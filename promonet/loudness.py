import multiprocessing as mp
import warnings

import librosa
import numpy as np
import torch

import promonet


###############################################################################
# Loudness feature
###############################################################################


def from_audio(audio):
    """Compute A-weighed loudness"""
    # Pad
    padding = (promonet.WINDOW_SIZE - promonet.HOPSIZE) // 2
    audio = torch.nn.functional.pad(
        audio[None],
        (padding, padding),
        mode='reflect').squeeze(0)

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

    # Average over weighted frequencies
    return torch.from_numpy(weighted.mean(axis=0)).float().to(device)[None]


def from_file(audio_file):
    """Compute A-weighed loudness from audio file"""
    return from_audio(promonet.load.audio(audio_file))


def from_file_to_file(audio_file, output_file):
    """Compute A-weighed loudness from audio file and save"""
    torch.save(from_file(audio_file), output_file)


def from_files_to_files(audio_files, output_files):
    """Compute A-weighed loudness from audio files and save"""
    with mp.get_context('spawn').Pool(promonet.NUM_WORKERS) as pool:
        pool.starmap(from_file_to_file, zip(audio_files, output_files))


###############################################################################
# Loudness utilities
###############################################################################


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


def normalize(loudness, min_db=promonet.MIN_DB, ref_db=promonet.REF_DB):
    """Normalize loudness to [-1., 1.]"""
    return (loudness - min_db) / (ref_db - min_db)


def perceptual_weights():
    """A-weighted frequency-dependent perceptual loudness weights"""
    frequencies = librosa.fft_frequencies(
        sr=promonet.SAMPLE_RATE,
        n_fft=promonet.WINDOW_SIZE)

    # A warning is raised for nearly inaudible frequencies, but it ends up
    # defaulting to -100 db. That default is fine for our purposes.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return librosa.A_weighting(frequencies)[:, None] - promonet.REF_DB


def scale(audio, target_loudness):
    """Scale the audio to the target loudness"""
    loudness = from_audio(audio.to(torch.float64))

    # Take difference and convert from dB to ratio
    gain = 10 ** ((target_loudness - loudness) / 20)

    # Linearly interpolate to the audio resolution
    gain = torch.nn.functional.interpolate(
        gain[None],
        size=audio.shape[1],
        mode='linear',
        align_corners=False)[0]

    # Scale
    scaled = gain * audio

    # Prevent clipping
    return limit(scaled)
