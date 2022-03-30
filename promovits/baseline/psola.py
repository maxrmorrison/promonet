# TEMPORARY - GPL license
import tempfile
from pathlib import Path

import torch
import psola
from parselmouth import Data, praat

import promovits


def from_audio(
    audio,
    sample_rate=promovits.SAMPLE_RATE,
    text=None,
    grid=None,
    target_loudness=None,
    target_pitch=None,
    checkpoint=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Performs pitch vocoding using Praat"""
    audio = audio.squeeze().numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Time-stretch
        if grid is not None:
            audio = time_stretch(audio, sample_rate, grid, tmpdir)

        # Pitch-shift
        if target_pitch is not None:

            # Maybe stretch pitch
            if grid is not None:
                target_pitch = promovits.interpolate.pitch(target_pitch, grid)

            audio = psola.core.pitch_shift(
                audio,
                target_pitch.squeeze().numpy(),
                promovits.FMIN,
                promovits.FMAX,
                sample_rate,
                tmpdir)

        # Convert to torch
        audio = torch.from_numpy(audio)[None]

        # Scale loudness
        if target_loudness is not None:
            audio = promovits.baseline.loudness.scale(audio, target_loudness)

        return audio


def time_stretch(audio, sample_rate, grid, tmpdir):
    """Perform praat time stretching on the manipulation"""
    # Get times in seconds
    times = promovits.HOPSIZE / promovits.SAMPLE_RATE * grid
    times = times.squeeze().numpy()

    # Get length of each output frame in input frames
    durations = grid[:, 1:] - grid[:, :-1]

    # Recover lost frame
    durations = torch.nn.functional.interpolate(
        durations[None].to(torch.float),
        durations.shape[1] + 1,
        mode='linear').squeeze().numpy()

    # Convert to ratio
    rates = (1. / durations)
    rates[rates < .0625] = .0625

    # Write duration to disk
    duration_file = str(tmpdir / 'duration.txt')
    psola.core.write_duration_tier(duration_file, times, rates)

    # Read duration file into praat
    duration_tier = Data.read(duration_file)

    # Open a praat manipulation context
    manipulation = psola.core.get_manipulation(
        audio,
        promovits.FMIN,
        promovits.FMAX,
        sample_rate,
        tmpdir)

    # Replace phoneme durations
    praat.call([duration_tier, manipulation], 'Replace duration tier')

    # Resynthesize
    return praat.call(manipulation, "Get resynthesis (overlap-add)").values[0]
