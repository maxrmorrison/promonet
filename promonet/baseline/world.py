import numpy as np
import pyworld
import scipy
import torch
import torchaudio
import torchutil

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
    pitch=None,
    periodicity=None
):
    """Perform World vocoding"""
    # Maybe resample
    if sample_rate != promonet.SAMPLE_RATE:
        resampler = torch.transforms.Resample(
            sample_rate,
            promonet.SAMPLE_RATE)
        audio = resampler(audio)

    # Get target number of frames
    if grid is not None:
        frames = grid.shape[-1]
    elif loudness is not None:
        frames = loudness.shape[-1]
    elif pitch is not None:
        frames = pitch.shape[-1]
    else:
        frames = promonet.convert.samples_to_frames(audio.shape[-1])

    # World parameterization
    target_pitch, spectrogram, aperiodicity = analyze(
        audio.squeeze().numpy(), frames)

    # Maybe time-stretch
    if grid is not None:
        (
            target_pitch,
            spectrogram,
            aperiodicity
        ) = linear_time_stretch(
            target_pitch,
            spectrogram,
            aperiodicity,
            grid.numpy()
        )

    # Maybe pitch-shift
    if pitch is not None:
        pitch = pitch.squeeze().numpy().astype(np.float64)

        # In WORLD, unvoiced frames are masked with zeros
        if periodicity is not None:
            unvoiced = \
                periodicity.squeeze().numpy() < promonet.VOICING_THRESHOLD
            pitch[unvoiced] = 0.
    else:
        pitch = target_pitch

    # Synthesize using modified parameters
    vocoded = pyworld.synthesize(
        pitch,
        spectrogram,
        aperiodicity,
        promonet.SAMPLE_RATE,
        promonet.HOPSIZE / promonet.SAMPLE_RATE * 1000.)

    # Convert to torch
    vocoded = torch.from_numpy(vocoded)[None]

    # Ensure correct length
    length = promonet.convert.frames_to_samples(len(pitch))
    if vocoded.shape[1] != length:
        temp = torch.zeros((1, length))
        crop_point = min(length, vocoded.shape[1])
        temp[:, :crop_point] = vocoded[:, :crop_point]
        vocoded = temp

    # Maybe scale loudness
    if loudness is not None:
        vocoded = promonet.preprocess.loudness.scale(
            vocoded,
            promonet.preprocess.loudness.band_average(loudness, 1))

    return vocoded.to(torch.float32)


def from_file(
    audio_file,
    grid_file=None,
    loudness_file=None,
    pitch_file=None,
    periodicity_file=None
):
    """Perform World vocoding on an audio file"""
    return from_audio(
        promonet.load.audio(audio_file),
        promonet.SAMPLE_RATE,
        None if grid_file is None else torch.load(grid_file),
        None if loudness_file is None else torch.load(loudness_file),
        None if pitch_file is None else torch.load(pitch_file),
        None if periodicity_file is None else torch.load(periodicity_file))


def from_file_to_file(
    audio_file,
    output_file,
    grid_file=None,
    loudness_file=None,
    pitch_file=None,
    periodicity_file=None
):
    """Perform World vocoding on an audio file and save"""
    vocoded = from_file(
        audio_file,
        grid_file,
        loudness_file,
        pitch_file,
        periodicity_file)
    torchaudio.save(output_file, vocoded, promonet.SAMPLE_RATE)


def from_files_to_files(
    audio_files,
    output_files,
    grid_files=None,
    loudness_files=None,
    pitch_files=None,
    periodicity_files=None
):
    """Perform World vocoding on multiple files and save"""
    if grid_files is None:
        grid_files = [None] * len(audio_files)
    if loudness_files is None:
        loudness_files = [None] * len(audio_files)
    if pitch_files is None:
        pitch_files = [None] * len(audio_files)
    if periodicity_files is None:
        periodicity_files = [None] * len(audio_files)
    iterator = zip(
        audio_files,
        output_files,
        grid_files,
        loudness_files,
        pitch_files,
        periodicity_files)
    for item in torchutil.iterator(iterator, 'world', total=len(audio_files)):
        from_file_to_file(*item)


###############################################################################
# Utilities
###############################################################################


def analyze(audio, frames):
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
    samples = promonet.convert.frames_to_samples(frames)
    pitch, time = pyworld.dio(
        audio,
        promonet.SAMPLE_RATE,
        frame_period=frame_period,
        f0_floor=promonet.FMIN,
        f0_ceil=promonet.FMAX,
        allowed_range=ALLOWED_RANGE)
    pitch = pitch[:frames]
    time = time[:frames]

    # Postprocess pitch
    pitch = pyworld.stonemask(audio, pitch, time, promonet.SAMPLE_RATE)

    # Extract spectrogram
    spectrogram = pyworld.cheaptrick(audio, pitch, time, promonet.SAMPLE_RATE)

    # Extract aperiodicity
    aperiodicity = pyworld.d4c(audio, pitch, time, promonet.SAMPLE_RATE)

    return pitch, spectrogram, aperiodicity


def linear_time_stretch(
    prev_pitch,
    prev_spectrogram,
    prev_aperiodicity,
    grid
):
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
        spectrogram[:, i] = np.interp(
            grid, prev_grid, prev_spectrogram[:, i])
        aperiodicity[:, i] = np.interp(
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
    pitch = 2 ** np.interp(grid, prev_grid, np.log2(pitch))

    # Apply time stretch to unvoiced sequence
    unvoiced = np.interp(grid, prev_grid, unvoiced)

    # Reapply unvoiced tokens
    pitch[unvoiced > .5] = 0.

    return pitch
