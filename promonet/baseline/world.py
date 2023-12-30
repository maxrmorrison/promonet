import numpy as np
import pyworld
import scipy
import torch

import promonet


###############################################################################
# Constants
###############################################################################


ALLOWED_RANGE = .8


###############################################################################
# World speech editing
###############################################################################


def from_audio(
    audio,
    sample_rate=promonet.SAMPLE_RATE,
    grid=None,
    loudness=None,
    pitch=None
):
    # Maybe resample
    if sample_rate != promonet.SAMPLE_RATE:
        resampler = torch.transforms.Resample(
            sample_rate,
            promonet.SAMPLE_RATE)
        audio = resampler(audio)

    # World parameterization
    audio = audio.squeeze().numpy()
    pitch, spectrogram, aperiodicity = analyze(audio)

    # Maybe time-stretch
    if grid is not None:
        pitch, spectrogram, aperiodicity = linear_time_stretch(
            pitch, spectrogram, aperiodicity, grid.numpy())

    # Maybe pitch-shift
    if pitch is not None:
        pitch = pitch.squeeze().numpy().astype(np.float64)
        # pitch[pitch != 0.] = pitch[pitch != 0.]

    # Synthesize using modified parameters
    vocoded = pyworld.synthesize(
        pitch,
        spectrogram,
        aperiodicity,
        promonet.SAMPLE_RATE,
        promonet.HOPSIZE / promonet.SAMPLE_RATE * 1000.)

    # Convert to torch
    vocoded = torch.from_numpy(vocoded)[None]

    # Maybe scale loudness
    if loudness is not None:
        vocoded = promonet.loudness.scale(vocoded, loudness)

    # Ensure correct length
    length = promonet.convert.frames_to_samples(len(pitch))
    if vocoded.shape[1] != length:
        temp = torch.zeros((1, length))
        crop_point = min(length, vocoded.shape[1])
        temp[:, :crop_point] = vocoded[:, :crop_point]
        vocoded = temp

    return vocoded.to(torch.float32)


def from_file(audio_file, grid_file=None, loudness_file=None, pitch_file=None):
    """Perform World vocoding on an audio file"""
    audio = promonet.load.audio(audio_file)
    return from_audio(
        audio,
        promonet.SAMPLE_RATE,
        grid_file,
        loudness_file,
        pitch_file)


def from_file_to_file(
    audio_file,
    output_file,
    grid_file=None,
    loudness_file=None,
    pitch_file=None
):
    """Perform World vocoding on an audio file and save"""
    vocoded = from_file(audio_file, grid_file, loudness_file, pitch_file)
    torch.save(vocoded, output_file)


def from_files_to_files(
    audio_files,
    output_files,
    grid_files=None,
    loudness_files=None,
    pitch_files=None
):
    """Perform World vocoding on multiple files and save"""
    torchutil.multiprocess_iterator(
        wrapper,
        zip(
            audio_files,
            output_files,
            grid_files,
            loudness_files,
            pitch_files
        ),
        'world',
        total=len(audio_files),
        num_workers=promonet.NUM_WORKERS)


###############################################################################
# Utilities
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
    frame_period = promonet.HOPSIZE / promonet.SAMPLE_RATE * 1000.

    # Extract pitch
    frames = promonet.convert.samples_to_frames(len(audio))
    samples = promonet.convert.frames_to_samples(frames)
    pitch, time = pyworld.dio(audio[:samples],
                              promonet.SAMPLE_RATE,
                              frame_period=frame_period,
                              f0_floor=promonet.FMIN,
                              f0_ceil=promonet.FMAX,
                              allowed_range=ALLOWED_RANGE)

    # Make sure number of frames is correct
    if len(pitch) > frames:
        difference = len(pitch) - frames
        pitch = pitch[:-difference]
        time = time[:-difference]

    # if len(pitch) != frames:
    #     import pdb; pdb.set_trace()
    #     prev_grid = np.arange(len(pitch))
    #     grid = np.linspace(0, len(pitch) - 1, frames)
    #     pitch = linear_time_stretch_pitch(pitch, prev_grid, grid, frames)
    #     time = scipy.interp(grid, prev_grid, time)

    # Postprocess pitch
    pitch = pyworld.stonemask(audio, pitch, time, promonet.SAMPLE_RATE)

    # Extract spectrogram
    spectrogram = pyworld.cheaptrick(audio, pitch, time, promonet.SAMPLE_RATE)

    # Extract aperiodicity
    aperiodicity = pyworld.d4c(audio, pitch, time, promonet.SAMPLE_RATE)

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


def wrapper(item):
    """Multiprocessing wrapper"""
    return from_file_to_file(*item)
