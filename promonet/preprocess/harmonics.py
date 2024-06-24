import os
from typing import List, Optional, Union

import librosa
import numpy as np
import penn
import scipy
import torbi
import torch
import torchaudio

import promonet


###############################################################################
# Extract harmonics
###############################################################################


def from_audio(
    audio: torch.Tensor,
    sample_rate: int = promonet.SAMPLE_RATE,
    pitch: Optional[torch.Tensor] = None,
    features: str = 'stft',
    decoder: str = 'viterbi',
    max_harmonics: int = promonet.MAX_HARMONICS,
    return_features: bool = False,
    gpu=None
) -> torch.Tensor:
    """Compute speech harmonic contours

    Arguments
        audio
            The speech recording
            shape=(1, samples)
        sample_rate
            The audio sampling rate
        pitch
            Optional pitch contour prior
        features
            The features to use for harmonic analysis.
            One of ['lpc', 'posteriorgram', 'stft'].
        decoder
            The decoding method. One of ['peak', 'viterbi'].
        max_harmonics
            The number of harmonics to compute
        return_features
            Whether to return the features used for analysis
        gpu
            The GPU index; defaults to CPU

    Returns
        Speech harmonics; NaNs indicate number of harmonics < max_harmonics
        shape=(max_harmonics, promonet.convert.samples_to_frames(samples))
    """
    # Preprocess
    if features == 'lpc':
        frames, frequencies = lpc_coefficients(audio, sample_rate)
    elif features == 'posteriorgram':
        frames, frequencies = pitch_posteriorgram(audio, sample_rate, gpu=gpu)
    elif features == 'stft':
        frames, frequencies = stft(audio, sample_rate, gpu=gpu)

    # Decode
    if decoder == 'peak':
        harmonics = peak_pick(frames, frequencies, max_harmonics)
    elif decoder == 'viterbi':
        harmonics = viterbi(
            frames,
            frequencies,
            pitch=pitch,
            max_harmonics=max_harmonics,
            gpu=gpu)

    if return_features:
        return harmonics, frames.T
    return harmonics


def from_file(
    file: Union[str, bytes, os.PathLike],
    pitch_file: Optional[Union[str, bytes, os.PathLike]] = None,
    max_harmonics: int = promonet.MAX_HARMONICS,
    return_features: bool = False,
    gpu=None
) -> torch.Tensor:
    """Compute speech harmonic contours from audio file


    Arguments
        file
            The speech audio file
        pitch_file
            Optional pitch contour prior
        max_harmonics
            The number of harmonics to compute
        return_features
            Whether to return the features used for analysis
        gpu
            The GPU index; defaults to CPU

    Returns
        Speech harmonics; NaNs indicate number of harmonics < max_harmonics
        shape=(max_harmonics, promonet.convert.samples_to_frames(samples))
    """
    pitch = None if pitch_file is None else torch.load(pitch_file)
    return from_audio(
        promonet.load.audio(file),
        pitch=pitch,
        max_harmonics=max_harmonics,
        return_features=return_features,
        gpu=gpu)


def from_file_to_file(
    file: Union[str, bytes, os.PathLike],
    output_file: Union[str, bytes, os.PathLike],
    pitch_file: Optional[Union[str, bytes, os.PathLike]] = None,
    output_feature_file: Optional[Union[str, bytes, os.PathLike]] = None,
    max_harmonics: int = promonet.MAX_HARMONICS,
    gpu=None
) -> None:
    """Compute speech harmonic contours from audio file and save


    Arguments
        file
            The speech audio file
        output_file
            PyTorch file to save results
        pitch_file
            Optional pitch contour prior
        output_feature_file
            Optional location to save the features used for analysis
        max_harmonics
            The number of harmonics to compute
        gpu
            The GPU index; defaults to CPU
    """
    result = from_file(
        file,
        pitch_file=pitch_file,
        max_harmonics=max_harmonics,
        return_features=output_feature_file is not None,
        gpu=gpu)
    if output_feature_file is not None:
        torch.save(result[-1].cpu(), output_feature_file)
    torch.save(result[0].cpu(), output_file)


def from_files_to_files(
    files: List[Union[str, bytes, os.PathLike]],
    output_files: List[Union[str, bytes, os.PathLike]],
    pitch_files: Optional[List[Union[str, bytes, os.PathLike]]] = None,
    output_feature_files: Optional[List[Union[str, bytes, os.PathLike]]] = None,
    max_harmonics: int = promonet.MAX_HARMONICS,
    gpu=None
) -> None:
    """Compute speech harmonic contours from audio files and save

    Arguments
        files
            The speech audio files
        output_files
            PyTorch files to save results
        pitch_files
            Optional pitch contour priors
        output_feature_files
            Optional locations to save the features used for analysis
        max_harmonics
            The number of harmonics to compute
        gpu
            The GPU index; defaults to CPU
    """
    if pitch_files is None:
        pitch_files = [None] * len(files)
    if output_feature_files is None:
        output_feature_files = [None] * len(files)
    for file, output_file, pitch_file, output_feature_file in zip(
        files,
        output_files,
        pitch_files,
        output_feature_files
    ):
        from_file_to_file(
            file,
            output_file,
            pitch_file,
            output_feature_file,
            max_harmonics,
            gpu=gpu)


###############################################################################
# Decode
###############################################################################


def peak_pick(frames, frequencies, max_harmonics=promonet.MAX_HARMONICS):
    """Decode harmonics via peak-picking"""
    # Find peaks
    peaks = [scipy.signal.find_peaks(frame)[0] for frame in frames]

    # Decode
    harmonics = torch.full((max_harmonics, len(frames)), float('nan'))
    for i, peak in enumerate(peaks):
        for j, p in enumerate(sorted(peak)):
            if j >= max_harmonics:
                continue
            harmonics[j, i] = frequencies[p]

    return harmonics


def viterbi(
    frames,
    frequencies,
    pitch=None,
    max_harmonics=promonet.MAX_HARMONICS,
    harmonic_width_ratio=0.8,
    gpu=None):
    """Decode harmonics via Viterbi decoding"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Normalize
    frames = frames.to(device)
    frequencies = frequencies.to(device)
    x = torch.clone(frames)
    x = torch.softmax(x + .5 * torch.arange(x.shape[-1], 0, -1, device=device), dim=1)

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
    initial = torch.linspace(1., 0., len(logfreq), device=device)
    initial /= initial.sum()

    # Maybe use more accurate external pitch esitimator for F0
    i = 0
    harmonics = torch.full((max_harmonics, len(x)), float('nan'), device=device)
    if pitch is not None:
        harmonics[0] = pitch.squeeze(0)
        i += 1

        # Mask
        x = torch.clone(frames)
        indices = torch.searchsorted(frequencies, harmonics[0])
        min_harmonic_idxs = torch.searchsorted(
            frequencies,
            harmonics[0] * (1. + harmonic_width_ratio))
        max_harmonic_idxs = torch.searchsorted(
            frequencies,
            harmonics[0] * (1. + 1. / harmonic_width_ratio))
        for j in range(len(indices)):
            x[j, :min_harmonic_idxs[j]] = -float('inf')
            x[j, max_harmonic_idxs[j]:] = -float('inf')
        x = torch.softmax(x, dim=1)

    # Iteratively decode F1, F2, ...
    while i < max_harmonics:

        # Decode
        indices = torbi.from_probabilities(
            x[None],
            transition=transition,
            initial=initial,
            log_probs=False,
            gpu=gpu
        )[0].to(torch.long)
        harmonics[i] = frequencies[indices]

        i += 1

        if i == max_harmonics:
            break

        # Mask
        x = torch.clone(frames)
        min_harmonic_idxs = torch.searchsorted(
            frequencies,
            harmonics[0] * (i + harmonic_width_ratio))
        max_harmonic_idxs = torch.searchsorted(
            frequencies,
            harmonics[0] * (i + 1. / harmonic_width_ratio))
        for j in range(len(indices)):
            x[j, :min_harmonic_idxs[j]] = -float('inf')
            x[j, max_harmonic_idxs[j]:] = -float('inf')
        x = torch.softmax(x, dim=1)

    return harmonics


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


def pitch_posteriorgram(audio, sample_rate, gpu=None):
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


def stft(
    audio,
    sample_rate=promonet.SAMPLE_RATE,
    fmin=promonet.FMIN,
    fmax=promonet.SAMPLE_RATE // 2,
    gpu=None
):
    """Compute short-time Fourier transform"""
    frames = promonet.convert.samples_to_frames(audio.shape[-1])

    # Device placement
    device = 'cpu' if gpu is None else f'cuda:{gpu}'
    audio = audio.to(device)

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
    num_fft = 4096
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
