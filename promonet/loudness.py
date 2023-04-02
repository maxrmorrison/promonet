import torch
import pysodic


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
