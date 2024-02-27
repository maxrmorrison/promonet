import os
from typing import List, Union

import librosa
import numpy as np
import scipy
import torch

import promonet


###############################################################################
# Extract formants
###############################################################################


def from_audio(
    audio: torch.Tensor,
    sample_rate: int = promonet.SAMPLE_RATE,
    max_formants: int = promonet.MAX_FORMANTS
) -> torch.Tensor:
    """Compute speech formant contours

    Arguments
        audio
            The speech recording
            shape=(1, samples)
        sample_rate
            The audio sampling rate
        max_formants
            The number of formants to compute

    Returns
        Speech formants; NaNs indicate number of formants < max_formants
        shape=(max_formants, promonet.convert.samples_to_frames(samples))
    """
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

    # Analyze
    peaks = []
    for frame in frames:
        a = librosa.lpc(frame.numpy(), int((sample_rate / 1000) + 2))
        _, h = scipy.signal.freqz([1], a, worN=len(frequencies))
        peak, _ = scipy.signal.find_peaks(np.log10(np.abs(h)))
        peaks.append(peak)

    # Decode
    formants = torch.full((max_formants, len(frames)), float('nan'))
    for i, peak in enumerate(peaks):
        for j, p in enumerate(sorted(peak)):
            if j >= max_formants:
                continue
            formants[j, i] = frequencies[p]

    return formants


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
