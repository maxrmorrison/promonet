# TEMPORARY - GPL license
import tempfile
from pathlib import Path

import psola
import torch
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
    audio = audio.squeeze().numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Time-stretch
        if grid is not None:
            audio = time_stretch(audio, sample_rate, grid, tmpdir)

        # Pitch-shift
        if pitch is not None:

            # Maybe stretch pitch
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
    torch.save(vocoded, output_file)


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
    # torchutil.multiprocess_iterator(
    #     wrapper,
    #     iterator,
    #     'psola',
    #     total=len(audio_files),
    #     num_workers=promonet.NUM_WORKERS)
    for item in torchutil.iterator(iterator, 'psola', total=len(audio_files)):
        from_file_to_file(*item)


###############################################################################
# Utilities
###############################################################################


def time_stretch(audio, sample_rate, grid, tmpdir):
    """Perform praat time stretching on the manipulation"""
    import pdb; pdb.set_trace()

    # Get times in seconds
    times = promonet.HOPSIZE / promonet.SAMPLE_RATE * grid
    times = times.squeeze().numpy()

    # Get length of each output frame in input frames
    durations = grid[1:] - grid[:-1]

    # Recover lost frame
    total = durations.sum().numpy()
    durations = torch.nn.functional.interpolate(
        durations[None, None].to(torch.float),
        len(durations) + 1,
        mode='linear',
        align_corners=False).squeeze().numpy()
    durations *= total / durations.sum()

    # Convert to ratio
    rates = (1. / durations)

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
    stretched = praat.call(
        manipulation,
        "Get resynthesis (overlap-add)").values[0]

    return stretched[:promonet.convert.frames_to_samples(len(rates))]


def wrapper(item):
    """Multiprocessing wrapper"""
    return from_file_to_file(*item)
