import pysodic
import torch

import promonet


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


def normalize(loudness, min_db=pysodic.MIN_DB, ref_db=pysodic.REF_DB):
    """Normalize loudness to [-1., 1.]"""
    return (loudness - min_db) / (ref_db - min_db)


def scale(audio, target_loudness):
    """Scale the audio to the target loudness"""
    loudness = pysodic.features.loudness(
        audio.to(torch.float64),
        promonet.SAMPLE_RATE,
        promonet.HOPSIZE / promonet.SAMPLE_RATE,
        promonet.WINDOW_SIZE / promonet.SAMPLE_RATE)

    # Maybe resample
    if loudness.shape[1] != target_loudness.shape[1]:
        loudness = torch.nn.functional.interpolate(
            loudness[None],
            target_loudness.shape[1],
            mode='linear',
            align_corners=False)[0]

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
