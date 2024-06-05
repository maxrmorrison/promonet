import functools
import multiprocessing as mp
import warnings

import librosa
import numpy as np
import torch

import promonet


###############################################################################
# Loudness feature
###############################################################################


def from_audio(audio, bands=1):
    """Compute A-weighted loudness"""
    # Pad
    padding = (promonet.WINDOW_SIZE - promonet.HOPSIZE) // 2
    audio = torch.nn.functional.pad(
        audio[None],
        (padding, padding),
        mode='reflect'
    ).squeeze(0)

    # Save device
    device = audio.device

    # Convert to numpy
    audio = audio.detach().cpu().numpy().squeeze(0)

    # Cache weights
    if not hasattr(from_audio, 'weights'):
        from_audio.weights = perceptual_weights()

    # Take stft
    stft = librosa.stft(
        audio,
        n_fft=promonet.WINDOW_SIZE,
        hop_length=promonet.HOPSIZE,
        win_length=promonet.WINDOW_SIZE,
        center=False)

    # Apply A-weighting in units of dB
    weighted = librosa.amplitude_to_db(np.abs(stft)) + from_audio.weights

    # Threshold
    weighted[weighted < promonet.MIN_DB] = promonet.MIN_DB

    # Multiband loudness
    loudness = torch.from_numpy(weighted).float().to(device)

    # Maybe average
    return band_average(loudness, bands) if bands is not None else loudness


def from_file(audio_file, bands=promonet.LOUDNESS_BANDS):
    """Compute A-weighted loudness from audio file"""
    return from_audio(promonet.load.audio(audio_file), bands)


def from_file_to_file(audio_file, output_file, bands=promonet.LOUDNESS_BANDS):
    """Compute A-weighted loudness from audio file and save"""
    torch.save(from_file(audio_file, bands), output_file)


def from_files_to_files(
    audio_files,
    output_files,
    bands=promonet.LOUDNESS_BANDS
):
    """Compute A-weighted loudness from audio files and save"""
    loudness_fn = functools.partial(from_file_to_file, bands=bands)
    with mp.get_context('spawn').Pool(promonet.NUM_WORKERS) as pool:
        pool.starmap(loudness_fn, zip(audio_files, output_files))


###############################################################################
# Loudness utilities
###############################################################################


def band_average(loudness, bands=promonet.LOUDNESS_BANDS):
    """Average over frequency bands"""
    if bands is not None:

        if bands == 1:

            # Average over all weighted frequencies
            loudness = loudness.mean(dim=-2, keepdim=True)

        else:

            # Average over loudness frequency bands
            step = loudness.shape[-2] / bands
            if loudness.ndim == 2:
                loudness = torch.stack(
                    [
                        loudness[int(band * step):int((band + 1) * step)].mean(dim=-2)
                        for band in range(int(bands))
                    ])
            else:
                loudness = torch.stack(
                    [
                        loudness[:, int(band * step):int((band + 1) * step)].mean(dim=-2)
                        for band in range(bands)
                    ],
                    dim=1)

    return loudness


def limit(audio, delay=40, attack_coef=.9, release_coef=.9995, threshold=.99):
    """Apply a limiter to prevent clipping"""
    # Delay compensation
    audio = torch.nn.functional.pad(audio, (0, delay - 1))

    current_gain = 1.
    delay_index = 0
    delay_line = torch.zeros(delay)
    envelope = 0

    for idx, sample in enumerate(audio[0]):

        # Update signal history
        delay_line[delay_index] = sample
        delay_index = (delay_index + 1) % delay

        # Calculate envelope
        envelope = max(abs(sample), envelope * release_coef)

        # Calcuate gain
        target_gain = threshold / envelope if envelope > threshold else 1.
        current_gain = \
            current_gain * attack_coef + target_gain * (1 - attack_coef)

        # Apply gain
        audio[:, idx] = delay_line[delay_index] * current_gain

    return audio[:, delay - 1:]


def normalize(loudness):
    """Normalize loudness to [-1., 1.]"""
    return (loudness - promonet.MIN_DB) / (promonet.REF_DB - promonet.MIN_DB)


def perceptual_weights():
    """A-weighted frequency-dependent perceptual loudness weights"""
    frequencies = librosa.fft_frequencies(
        sr=promonet.SAMPLE_RATE,
        n_fft=promonet.WINDOW_SIZE)

    # A warning is raised for nearly inaudible frequencies, but it ends up
    # defaulting to -100 db. That default is fine for our purposes.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return (
            librosa.A_weighting(frequencies)[:, None] - float(promonet.REF_DB))


def scale(audio, target_loudness):
    """Scale the audio to the target loudness"""
    # Maybe average to get scalar loudness
    if target_loudness.shape[-2] > 1:
        target_loudness = target_loudness.mean(dim=-2, keepdim=True)

    # Get current loudness
    loudness = from_audio(audio.to(torch.float64))

    # Take difference and convert from dB to ratio
    gain = promonet.convert.db_to_ratio(target_loudness - loudness)

    # Apply gain and prevent clipping
    return limit(shift(audio, gain))


def shift(audio, value):
    """Shift loudness by target value in decibels"""
    # Convert from dB to ratio
    gain = promonet.convert.db_to_ratio(value)

    # Linearly interpolate to the audio resolution
    if isinstance(gain, torch.Tensor) and gain.numel() > 1:
        gain = torch.nn.functional.interpolate(
            gain[None],
            size=audio.shape[1],
            mode='linear',
            align_corners=False)[0]

    # Scale
    return gain * audio
