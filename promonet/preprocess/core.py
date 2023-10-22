import functools
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import ppgs
import pyfoal
import pypar
import pysodic
import torch
import torchaudio

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
    # Infer prosody
    pitch, periodicity, loudness = pysodic.pitch_periodicity_loudness(
        audio,
        sample_rate,
        promonet.convert.samples_to_seconds(promonet.HOPSIZE),
        promonet.convert.samples_to_seconds(promonet.WINDOW_SIZE),
        gpu=gpu)

    # Infer ppg
    ppg = ppgs.from_audio(audio, sample_rate, gpu=gpu)
    grid = promonet.interpolate.grid.of_length(ppg, pitch.shape[-1])
    ppg = promonet.interpolate.ppg(ppg, grid)

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
    torch.save(ppg, f'{output_prefix}-ppg.pt')


def from_files_to_files(
    files: List[Union[str, bytes, os.PathLike]],
    output_prefixes: Optional[List[Union[str, os.PathLike]]] = None,
    features=promonet.INPUT_FEATURES,
    gpu=None
) -> None:
    """Preprocess multiple audio files on disk and save

    Arguments
        files: Audio files to preprocess
        output_prefixes: Files to save features, minus extension
        features: The features to preprocess
        gpu: The GPU index
    """
    if output_prefixes is None:
        output_prefixes = [file.parent / file.stem for file in files]

    # Preprocess phonetic posteriorgrams
    if 'ppg' in features:
        ppgs.from_files_to_files(
            files,
            [f'{prefix}-ppg.pt' for prefix in output_prefixes],
            num_workers=promonet.NUM_WORKERS,
            gpu=gpu)

    # Preprocess prosody features
    if any(feature in features for feature in [
        'loudness',
        'periodicity',
        'pitch'
    ]):
        for file, prefix in promonet.iterator(
            zip(files, output_prefixes),
            'pysodic',
            total=len(files)
        ):
            pysodic.from_file_to_file(
                file,
                prefix,
                hopsize=promonet.HOPSIZE / promonet.SAMPLE_RATE,
                window_size=promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
                voicing_threshold=0.1625,
                gpu=gpu)


###############################################################################
# Utilities
###############################################################################


def forced_alignment(text_files, audio_files, output_prefixes):
    """Compute forced phoneme alignments"""
    if output_prefixes is None:
        output_prefixes = [file.parent / file.stem for file in text_files]

    # Output files
    alignment_files = [
        Path(f'{prefix}.TextGrid') for prefix in output_prefixes]

    # Default to using all cpus
    num_workers = max(min(len(text_files) // 2, promonet.NUM_WORKERS), 1)

    # Launch multiprocessed P2FA alignment
    align_fn = functools.partial(pyfoal_catch_failed)
    iterator = zip(text_files, audio_files, alignment_files)
    with mp.get_context('spawn').Pool(num_workers) as pool:
        failed = pool.starmap(align_fn, iterator)

    # Handle alignment failures of data-augmented speech by stretching
    # non-augmented alignments
    for file in [i for i in failed if i is not None]:
        good_textgrid = file.parent / f'{file.stem[:-4]}-100.TextGrid'
        ratio = int(file.stem[-3:]) / 100.
        alignment = pypar.Alignment(good_textgrid)
        durations = [
            phoneme.duration() for phoneme in alignment.phonemes()]
        new_durations = [duration / ratio for duration in durations]
        alignment.update(durations=new_durations)
        alignment.save(file.parent / f'{file.stem}.TextGrid')

    # Get exact lengths derived from audio files to avoid length
    # mismatch due to floating-point vs integer hopsize
    hopsize = promonet.HOPSIZE / promonet.SAMPLE_RATE
    lengths = [
        int(
            torchaudio.info(file).num_frames //
            (torchaudio.info(file).sample_rate * hopsize)
        )
        for file in audio_files]

    # Convert alignments to indices
    indices_files = [
        file.parent / f'{file.stem}-phonemes.pt'
        for file in alignment_files]
    pysodic.alignment_files_to_indices_files(
        alignment_files,
        indices_files,
        lengths,
        hopsize)


def pyfoal_catch_failed(text_file, audio_file, output_file):
    try:
        pyfoal.from_file_to_file(
            text_file,
            audio_file,
            output_file,
            aligner='p2fa')
    except IndexError:
        return audio_file
