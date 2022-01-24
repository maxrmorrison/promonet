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
        pitch = target_pitch.squeeze().numpy().astype(np.float64)

    # Maybe time-stretch
    if grid is not None:
        pitch, spectrogram, aperiodicity = linear_time_stretch(
            pitch, spectrogram, aperiodicity, grid)

    # Synthesize using modified parameters
    vocoded = pyworld.synthesize(
        pitch,
        spectrogram,
        aperiodicity,
        promovits.SAMPLE_RATE,
        promovits.HOPSIZE / promovits.SAMPLE_RATE * 1000.)

    # Trim zero padding
    return vocoded


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

    # Pitch
    pitch, time = pyworld.dio(audio,
                              promovits.SAMPLE_RATE,
                              frame_period=frame_period,
                              f0_floor=promovits.FMIN,
                              f0_ceil=promovits.FMAX,
                              allowed_range=ALLOWED_RANGE)
    pitch = pyworld.stonemask(audio, pitch, time, promovits.SAMPLE_RATE)

    # Spectrogram
    spectrogram = pyworld.cheaptrick(audio, pitch, time, promovits.SAMPLE_RATE)

    # Aperiodicity
    aperiodicity = pyworld.d4c(audio, pitch, time, promovits.SAMPLE_RATE)

    return pitch, spectrogram, aperiodicity


# TODO - apply grid to features
def linear_time_stretch(prev_pitch,
                        prev_spectrogram,
                        prev_aperiodicity,
                        duration):
    """Apply time stretch in WORLD parameter space

    Arguments
        prev_pitch : np.array(shape=(frames,))
            The pitch to be stretched
        prev_spectrogram : np.array(shape=(frames, frequencies))
            The spectrogram to be stretched
        prev_aperiodicity : np.array(shape=(frames, frequencies))
            The aperiodicity to be stretched
        duration : float
            The new duration in seconds
    """
    # Number of frames before and after
    prev_frames = len(prev_pitch)
    next_frames = clpcnet.convert.seconds_to_frames(duration)

    # Time-aligned grid before and after
    prev_grid = np.linspace(0, prev_frames - 1, prev_frames)
    next_grid = np.linspace(0, prev_frames - 1, next_frames)

    # Apply time stretch to pitch
    pitch = linear_time_stretch_pitch(
        prev_pitch, prev_grid, next_grid, next_frames)

    # Allocate spectrogram and aperiodicity buffers
    frequencies = prev_spectrogram.shape[1]
    spectrogram = np.zeros((next_frames, frequencies))
    aperiodicity = np.zeros((next_frames, frequencies))

    # Apply time stretch to all channels of spectrogram and aperiodicity
    for i in range(frequencies):
        spectrogram[:, i] = scipy.interp(
            next_grid, prev_grid, prev_spectrogram[:, i])
        aperiodicity[:, i] = scipy.interp(
            next_grid, prev_grid, prev_aperiodicity[:, i])

    return pitch, spectrogram, aperiodicity


def linear_time_stretch_pitch(pitch, prev_grid, next_grid, next_frames):
    """Perform time-stretching on pitch features"""
    if (pitch == 0.).all():
        return np.zeros(next_frames)

    # Get unvoiced tokens
    unvoiced = pitch == 0.

    # Linearly interpolate unvoiced regions
    pitch[unvoiced] = np.interp(
        np.where(unvoiced)[0], np.where(~unvoiced)[0], pitch[~unvoiced])

    # Apply time stretch to pitch
    pitch = scipy.interp(next_grid, prev_grid, pitch)

    # Apply time stretch to unvoiced sequence
    unvoiced = scipy.interp(next_grid, prev_grid, unvoiced)

    # Reapply unvoiced tokens
    pitch[unvoiced > .5] = 0.

    return pitch
