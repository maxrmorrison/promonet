# TEMPORARY - GPL license
import tempfile
from pathlib import Path

import numpy as np
import psola
import torch
import torchaudio
import torchutil
from parselmouth import Data, praat

import promonet


###############################################################################
# PSOLA speech editing
###############################################################################


def from_audio(
    audio,
    sample_rate=promonet.SAMPLE_RATE,
    grid=None,
    loudness=None,
    pitch=None
):
    """Performs speech editing using Praat"""
    if grid is None:
        expected_frames = promonet.convert.samples_to_frames(audio.shape[1])
    else:
        expected_frames = grid.shape[0]

    audio = audio.squeeze().numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Time-stretch
        if grid is not None:
            audio = time_stretch(audio, sample_rate, grid, tmpdir)

        # Pitch-shift
        if pitch is not None:
            if grid is not None:
                pitch = 2 ** promonet.edit.grid.sample(torch.log2(pitch), grid)
            audio = psola.core.pitch_shift(
                audio,
                pitch.squeeze().numpy(),
                promonet.FMIN,
                promonet.FMAX,
                sample_rate,
                tmpdir)

        # Convert to torch
        audio = torch.from_numpy(audio)[None].to(torch.float32)

        # Scale loudness
        if loudness is not None:
            audio = promonet.loudness.scale(audio, loudness)

        # Maybe pad or trim
        frames = promonet.convert.samples_to_frames(audio.shape[1])
        if frames == expected_frames + 1:
            audio = audio[:, 128:-128]
        elif frames == expected_frames - 1:
            audio = torch.nn.functional.pad(
                audio[None],
                (promonet.HOPSIZE // 2, promonet.HOPSIZE // 2),
                mode='reflect'
            ).squeeze(0)

        return audio


def from_file(audio_file, grid_file=None, loudness_file=None, pitch_file=None):
    """Perform PSOLA vocoding on an audio file"""
    return from_audio(
        promonet.load.audio(audio_file),
        promonet.SAMPLE_RATE,
        None if grid_file is None else torch.load(grid_file),
        None if loudness_file is None else torch.load(loudness_file),
        None if pitch_file is None else torch.load(pitch_file))


def from_file_to_file(
    audio_file,
    output_file,
    grid_file=None,
    loudness_file=None,
    pitch_file=None
):
    """Perform PSOLA vocoding on an audio file and save"""
    vocoded = from_file(audio_file, grid_file, loudness_file, pitch_file)
    torchaudio.save(output_file, vocoded, promonet.SAMPLE_RATE)


def from_files_to_files(
    audio_files,
    output_files,
    grid_files=None,
    loudness_files=None,
    pitch_files=None
):
    """Perform PSOLA vocoding on multiple files and save"""
    if grid_files is None:
        grid_files = [None] * len(audio_files)
    if loudness_files is None:
        loudness_files = [None] * len(audio_files)
    if pitch_files is None:
        pitch_files = [None] * len(audio_files)
    iterator = zip(
        audio_files,
        output_files,
        grid_files,
        loudness_files,
        pitch_files)
    for item in torchutil.iterator(iterator, 'psola', total=len(audio_files)):
        from_file_to_file(*item)


###############################################################################
# Utilities
###############################################################################


def time_stretch(audio, sample_rate, grid, tmpdir):
    """Perform praat time stretching on the manipulation"""
    # Interpolate grid
    grid = torch.nn.functional.interpolate(
        grid[None, None],
        len(grid) + 1,
        mode='linear',
        align_corners=False
    )[0, 0].numpy()

    # Get times in seconds
    times = promonet.convert.frames_to_seconds(grid)

    # Get stretch ratio
    rates = 1. / (grid[1:] - grid[:-1])

    # Write duration to disk
    duration_file = str(tmpdir / 'duration.txt')
    psola.core.write_duration_tier(duration_file, times, rates)

    # Read duration file into praat
    duration_tier = Data.read(duration_file)

    # Open a praat manipulation context
    manipulation = psola.core.get_manipulation(
        audio,
        promonet.FMIN,
        promonet.FMAX,
        sample_rate,
        tmpdir)

    # Replace phoneme durations
    praat.call([duration_tier, manipulation], 'Replace duration tier')

    # Resynthesize
    return praat.call(manipulation, 'Get resynthesis (overlap-add)').values[0]
