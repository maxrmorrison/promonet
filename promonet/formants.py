import os
from typing import List, Union

import librosa
import numpy as np
import penn
import scipy
import torbi
import torch
import torchaudio

import promonet


###############################################################################
# Extract formants
###############################################################################


def from_audio(
    audio: torch.Tensor,
    sample_rate: int = promonet.SAMPLE_RATE,
    features: str = 'stft',
    decoder: str = 'viterbi',
    max_formants: int = promonet.MAX_FORMANTS
) -> torch.Tensor:
    """Compute speech formant contours

    Arguments
        audio
            The speech recording
            shape=(1, samples)
        sample_rate
            The audio sampling rate
        features
            The features to use for formant analysis.
            One of ['lpc', 'posteriorgram', 'stft'].
        decider
            The decoding method. One of ['peak', 'viterbi'].
        max_formants
            The number of formants to compute

    Returns
        Speech formants; NaNs indicate number of formants < max_formants
        shape=(max_formants, promonet.convert.samples_to_frames(samples))
    """
    # Preprocess
    if features == 'lpc':
        frames, frequencies = lpc_coefficients(audio, sample_rate)
    elif features == 'posteriorgram':
        frames, frequencies = pitch_posteriorgram(audio, sample_rate)
    elif features == 'stft':
        frames, frequencies = stft(audio, sample_rate)

    # Decode
    if decoder == 'peak':
        return peak_pick(frames, frequencies, max_formants)
    elif decoder == 'viterbi':
        return viterbi(frames, frequencies, max_formants)


def from_file(
    file: Union[str, bytes, os.PathLike],
    max_formants: int = promonet.MAX_FORMANTS
) -> torch.Tensor:
    """Compute speech formant contours from audio file


    Arguments
        file
            The speech audio file
        max_formants
            The number of formants to compute

    Returns
        Speech formants; NaNs indicate number of formants < max_formants
        shape=(max_formants, promonet.convert.samples_to_frames(samples))
    """
    return from_audio(promonet.load.audio(file), max_formants=max_formants)


def from_file_to_file(
    file: Union[str, bytes, os.PathLike],
    output_file: Union[str, bytes, os.PathLike],
    max_formants: int = promonet.MAX_FORMANTS
) -> None:
    """Compute speech formant contours from audio file and save


    Arguments
        file
            The speech audio file
        output_file
            PyTorch file to save results
        max_formants
            The number of formants to compute
    """
    torch.save(from_file(file, max_formants), output_file)


def from_files_to_files(
    files: List[Union[str, bytes, os.PathLike]],
    output_files: List[Union[str, bytes, os.PathLike]],
    max_formants: int = promonet.MAX_FORMANTS
) -> None:
    """Compute speech formant contours from audio files and save

    Arguments
        files
            The speech audio files
        output_files
            PyTorch files to save results
        max_formants
            The number of formants to compute
    """
    for file, output_file in zip(files, output_files):
        from_file_to_file(file, output_file, max_formants)


###############################################################################
# Decode
###############################################################################


def peak_pick(frames, frequencies, max_formants=promonet.MAX_FORMANTS):
    """Decode formants via peak-picking"""
    # Find peaks
    peaks = [scipy.signal.find_peaks(frame)[0] for frame in frames]

    # Decode
    formants = torch.full((max_formants, len(frames)), float('nan'))
    for i, peak in enumerate(peaks):
        for j, p in enumerate(sorted(peak)):
            if j >= max_formants:
                continue
            formants[j, i] = frequencies[p]

    return formants


def viterbi(
    frames,
    frequencies,
    max_formants=promonet.MAX_FORMANTS,
    min_formant_width=14,
    max_formant_width=30):
    """Decode formants via Viterbi decoding"""
    # Normalize
    x = torch.clone(frames)
    x = torch.softmax(x + .5 * torch.arange(x.shape[-1], 0, -1), dim=1)

    # Transition matrix
    logfreq = torch.log2(frequencies)
    transition = 1. - 3.5 * torch.cdist(
        logfreq[None, :, None],
        logfreq[None, :, None],
        p=1.0
    )[0]
    transition[transition < 0.] = 0.
    transition /= transition.sum(dim=1)

    # Initial matrix
    initial = torch.linspace(1., 0., len(logfreq))
    initial /= initial.sum()

    # Iteratively decode F0, F1, F2, ...
    formants = torch.full((max_formants, len(x)), float('nan'))
    for i in range(len(formants)):

        # Decode
        indices = torbi.from_probabilities(
            x[None],
            transition=transition,
            initial=initial,
            log_probs=False
        )[0].to(torch.long)
        formants[i] = frequencies[indices]

        # Mask
        x = torch.clone(frames)
        for j, idx in enumerate(indices):
            x[j, :idx + min_formant_width] = -float('inf')
            x[j, idx + min_formant_width + max_formant_width:] = -float('inf')
        x = torch.softmax(x, dim=1)

    return formants


###############################################################################
# Preprocess
###############################################################################


def lpc_coefficients(audio, sample_rate):
    """Compute linear predictive coding coefficients"""
    # Pad
    padding = (promonet.WINDOW_SIZE - promonet.HOPSIZE) // 2
    audio = torch.nn.functional.pad(audio, (padding, padding), 'constant')

    # Chunk
    frames = torch.nn.functional.unfold(
        audio[:, None, None],
        kernel_size=(1, promonet.WINDOW_SIZE),
        stride=(1, promonet.HOPSIZE))[0]

    # Window
    frames = frames.T * torch.hamming_window(promonet.WINDOW_SIZE)

    # Real FFT frequencies in Hz
    frequencies = sample_rate * torch.linspace(0., 1., promonet.NUM_FFT)
    frequencies = frequencies[0:len(frequencies) // 2]

    result = []
    for frame in frames:
        a = librosa.lpc(frame.numpy(), int((sample_rate / 1000) + 2))
        _, h = scipy.signal.freqz([1], a, worN=len(frequencies))
        result.append(torch.tensor(np.log10(np.abs(h))))

    return torch.stack(result, dim=0), frequencies


def pitch_posteriorgram(audio, sample_rate):
    """Compute pitch posteriorgram"""
    result = []

    # Preprocess audio
    for frames in penn.preprocess(
        audio,
        sample_rate,
        hopsize=promonet.convert.samples_to_seconds(promonet.HOPSIZE),
        center='half-hop'
    ):

        # Infer
        result.append(penn.infer(frames).detach().squeeze(2))

    # Concatenate
    result = torch.cat(result, 0)

    # Mask extrema
    minidx = penn.convert.frequency_to_bins(torch.tensor(50.))
    maxidx = penn.convert.frequency_to_bins(torch.tensor(1600.))
    result[:, :minidx] = -float('inf')
    result[:, maxidx:] = -float('inf')

    # Frequency bins
    frequencies = penn.convert.bins_to_frequency(torch.arange(penn.PITCH_BINS))

    return result, frequencies


def stft(audio, sample_rate=promonet.SAMPLE_RATE, fmin=promonet.FMIN, fmax=promonet.SAMPLE_RATE // 2):
    """Compute short-time Fourier transform"""
    frames = promonet.convert.samples_to_frames(audio.shape[-1])

    # Low-pass filter to remove low frequencies
    audio = torchaudio.functional.highpass_biquad(
        audio,
        sample_rate,
        1.33 * fmin)

    # Resample to 4 kHz to remove upper harmonics
    target_sample_rate = 2 * fmax
    audio = torchaudio.functional.resample(
        audio,
        sample_rate,
        target_sample_rate)

    # Pad audio
    num_fft = 4 * promonet.NUM_FFT
    hopsize = int(promonet.HOPSIZE * target_sample_rate / sample_rate)
    size = (
        hopsize * (frames - (audio.shape[-1] // hopsize)) // 2 +
        (num_fft - promonet.HOPSIZE) // 2)
    audio = torch.nn.functional.pad(audio, (size, size), mode='reflect')

    # Compute STFT
    window = torch.hann_window(
        num_fft,
        dtype=audio.dtype,
        device=audio.device)

    # Compute stft
    stft = torch.stft(
        audio.squeeze(1),
        num_fft,
        hop_length=hopsize,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True)
    stft = torch.view_as_real(stft)

    # Compute magnitude
    spectrogram = torch.sqrt(stft.pow(2).sum(-1) + 1e-6)

    # Compute STFT frequencies
    frequencies = torch.abs(torch.fft.fftfreq(
        num_fft,
        1 / target_sample_rate
    )[:num_fft // 2 + 1])

    # Crop off frequencies below threshold
    minidx = torch.searchsorted(frequencies, torch.tensor(fmin))

    return spectrogram.squeeze(0)[minidx:].T, frequencies[minidx:]
