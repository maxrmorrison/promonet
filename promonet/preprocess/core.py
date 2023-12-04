import os
from typing import List, Optional, Tuple, Union

import penn
import ppgs
import torch

import promonet


###############################################################################
# Preprocess
###############################################################################


def from_audio(
    audio: torch.Tensor,
    sample_rate: int = promonet.SAMPLE_RATE,
    gpu: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess audio

    Arguments
        audio: Audio to preprocess
        sample_rate: Audio sample rate
        gpu: The GPU index

    Returns
        pitch: The pitch contour
        periodicity: The periodicity contour
        loudness: The loudness contour
        ppg: The phonetic posteriorgram
    """
    # Estimate pitch and periodicity
    pitch, periodicity = penn.from_audio(
        audio,
        sample_rate=sample_rate,
        hopsize=promonet.convert.samples_to_seconds(promonet.HOPSIZE),
        fmin=promonet.FMIN,
        fmax=promonet.FMAX,
        batch_size=2048,
        center='half-hop',
        interp_unvoiced_at=promonet.VOICING_THRESOLD,
        gpu=gpu)

    # Compute loudness
    loudness = promonet.loudness.from_audio(audio).to(pitch.device)

    # Infer ppg
    ppg = ppgs.from_audio(audio, sample_rate, gpu=gpu)
    ppg = promonet.load.ppg(ppg, resample_length=pitch.shape[-1])

    return pitch, periodicity, loudness, ppg


def from_file(
    file: Union[str, bytes, os.PathLike],
    gpu: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess audio on disk

    Arguments
        file: Audio file to preprocess
        gpu: The GPU index

    Returns
        pitch: The pitch contour
        periodicity: The periodicity contour
        loudness: The loudness contour
        ppg: The phonetic posteriorgram
    """
    return from_audio(promonet.load.audio(file), gpu=gpu)


def from_file_to_file(
    file: Union[str, bytes, os.PathLike],
    output_prefix: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> None:
    """Preprocess audio on disk and save

    Arguments
        file: Audio file to preprocess
        output_prefix: File to save features, minus extension
        gpu: The GPU index
    """
    # Preprocess
    pitch, periodicity, loudness, ppg = from_file(file, gpu)

    # Save
    if output_prefix is None:
        output_prefix = file.parent / file.stem
    torch.save(pitch, f'{output_prefix}-pitch.pt')
    torch.save(periodicity, f'{output_prefix}-periodicity.pt')
    torch.save(loudness, f'{output_prefix}-loudness.pt')
    torch.save(ppg, f'{output_prefix}{ppgs.representation_output_extension()}')


def from_files_to_files(
    files: List[Union[str, bytes, os.PathLike]],
    output_prefixes: Optional[List[Union[str, os.PathLike]]] = None,
    gpu=None
) -> None:
    """Preprocess multiple audio files on disk and save

    Arguments
        files: Audio files to preprocess
        output_prefixes: Files to save features, minus extension
        gpu: The GPU index
    """
    if output_prefixes is None:
        output_prefixes = [file.parent / file.stem for file in files]

    # Preprocess phonetic posteriorgrams
    extension = ppgs.representation_file_extension()
    ppgs.from_files_to_files(
        files,
        [f'{prefix}{extension}' for prefix in output_prefixes],
        num_workers=promonet.NUM_WORKERS,
        gpu=gpu)

    # Preprocess pitch and periodicity
    penn.from_files_to_files(
        files,
        output_prefixes,
        hopsize=promonet.convert.samples_to_seconds(promonet.HOPSIZE),
        fmin=promonet.FMIN,
        fmax=promonet.FMAX,
        batch_size=2048,
        center='half-hop',
        interp_unvoiced_at=promonet.VOICING_THRESOLD,
        gpu=gpu)

    # Preprocess loudness
    promonet.loudness.from_files_to_files(
        files,
        [f'{prefix}-loudness.pt' for prefix in output_prefixes])
