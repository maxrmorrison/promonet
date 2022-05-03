import numpy as np
import pyworld
import scipy
import torch

import promovits


###############################################################################
# WORLD constants
###############################################################################


ALLOWED_RANGE = .8


###############################################################################
# Pitch-shifting and time-stretching with WORLD
###############################################################################


def from_audio(
    audio,
    sample_rate=promovits.SAMPLE_RATE,
    text=None,
    grid=None,
    target_loudness=None,
    target_pitch=None,
    checkpoint=None,
    gpu=None):
    # Maybe resample
    if sample_rate != promovits.SAMPLE_RATE:
        resampler = torch.transforms.Resample(
            sample_rate,
            promovits.SAMPLE_RATE)
        audio = resampler(audio)

    # World parameterization
    audio = audio.squeeze().numpy()
    pitch, spectrogram, aperiodicity = analyze(audio)

    # Maybe pitch-shift
    if target_pitch is not None:
        target_pitch = target_pitch.squeeze().numpy().astype(np.float64)
        pitch[pitch != 0.] = target_pitch[pitch != 0.]

    # Maybe time-stretch
    if grid is not None:
        pitch, spectrogram, aperiodicity = linear_time_stretch(
            pitch, spectrogram, aperiodicity, grid.numpy())

    # Synthesize using modified parameters
    vocoded = pyworld.synthesize(
        pitch,
        spectrogram,
        aperiodicity,
        promovits.SAMPLE_RATE,
        promovits.HOPSIZE / promovits.SAMPLE_RATE * 1000.)

    # Convert to torch
    vocoded = torch.from_numpy(vocoded)[None]

    # Maybe scale loudness
    if target_loudness is not None:
        vocoded = promovits.baseline.loudness.scale(vocoded, target_loudness)

    # Ensure correct length
    length = len(pitch) * promovits.HOPSIZE
    if vocoded.shape[1] != length:
        temp = torch.zeros((1, length))
        crop_point = min(length, vocoded.shape[1])
        temp[:, :crop_point] = vocoded[:, :crop_point]
        vocoded = temp

    return vocoded.to(torch.float32)


###############################################################################
# Vocoding utilities
###############################################################################


def analyze(audio):
    """Convert an audio signal to WORLD parameter representation
    Arguments
        audio : np.array(shape=(samples,))
            The audio being analyzed
    Returns
        pitch : np.array(shape=(frames,))
            The pitch contour
        spectrogram : np.array(shape=(frames, channels))
            The audio spectrogram
        aperiodicity : np.array(shape=(frames,))
            The voiced/unvoiced confidence
    """
    # Cast to double
    audio = audio.astype(np.float64)

    # Hopsize in milliseconds
    frame_period = promovits.HOPSIZE / promovits.SAMPLE_RATE * 1000.

    # Extract pitch
    pitch, time = pyworld.dio(audio,
                              promovits.SAMPLE_RATE,
                              frame_period=frame_period,
                              f0_floor=promovits.FMIN,
                              f0_ceil=promovits.FMAX,
                              allowed_range=ALLOWED_RANGE)

    # Make sure number of frames is correct
    frames = len(audio) // promovits.HOPSIZE
    if len(pitch) != frames:
        prev_grid = np.arange(len(pitch))
        grid = np.linspace(0, len(pitch) - 1, frames)
        pitch = linear_time_stretch_pitch(pitch, prev_grid, grid, frames)
        time = scipy.interp(grid, prev_grid, time)

    # Postprocess pitch
    pitch = pyworld.stonemask(audio, pitch, time, promovits.SAMPLE_RATE)

    # Extract spectrogram
    spectrogram = pyworld.cheaptrick(audio, pitch, time, promovits.SAMPLE_RATE)

    # Extract aperiodicity
    aperiodicity = pyworld.d4c(audio, pitch, time, promovits.SAMPLE_RATE)

    return pitch, spectrogram, aperiodicity


def linear_time_stretch(prev_pitch,
                        prev_spectrogram,
                        prev_aperiodicity,
                        grid):
    """Apply time stretch in WORLD parameter space"""
    grid = grid[0] if grid.ndim == 2 else grid

    # Number of frames before and after
    prev_frames = len(prev_pitch)
    next_frames = len(grid)

    # Time-aligned grid before and after
    prev_grid = np.linspace(0, prev_frames - 1, prev_frames)

    # Apply time stretch to pitch
    pitch = linear_time_stretch_pitch(
        prev_pitch, prev_grid, grid, next_frames)

    # Allocate spectrogram and aperiodicity buffers
    frequencies = prev_spectrogram.shape[1]
    spectrogram = np.zeros((next_frames, frequencies))
    aperiodicity = np.zeros((next_frames, frequencies))

    # Apply time stretch to all channels of spectrogram and aperiodicity
    for i in range(frequencies):
        spectrogram[:, i] = scipy.interp(
            grid, prev_grid, prev_spectrogram[:, i])
        aperiodicity[:, i] = scipy.interp(
            grid, prev_grid, prev_aperiodicity[:, i])

    return pitch, spectrogram, aperiodicity


def linear_time_stretch_pitch(pitch, prev_grid, grid, next_frames):
    """Perform time-stretching on pitch features"""
    if (pitch == 0.).all():
        return np.zeros(next_frames)

    # Get unvoiced tokens
    unvoiced = pitch == 0.

    # Linearly interpolate unvoiced regions
    pitch[unvoiced] = np.interp(
        np.where(unvoiced)[0], np.where(~unvoiced)[0], pitch[~unvoiced])

    # Apply time stretch to pitch in base-2 log-space
    pitch = 2 ** scipy.interp(grid, prev_grid, np.log2(pitch))

    # Apply time stretch to unvoiced sequence
    unvoiced = scipy.interp(grid, prev_grid, unvoiced)

    # Reapply unvoiced tokens
    pitch[unvoiced > .5] = 0.

    return pitch
