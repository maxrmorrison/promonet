import pysodic
import torch

import promovits


def scale(audio, target_loudness):
    """Scale the audio to the target loudness"""
    loudness = pysodic.features.loudness(
        audio,
        promovits.SAMPLE_RATE,
        promovits.HOPSIZE / promovits.SAMPLE_RATE,
        promovits.WINDOW_SIZE / promovits.SAMPLE_RATE)

    # Take different and convert from dB to ratio
    gain = 10 ** ((target_loudness - loudness) / 20)

    # Linearly interpolate to the audio resolution
    gain = torch.nn.functional.interpolate(
        gain[None],
        size=audio.shape[1],
        mode='linear')[0]

    # Scale
    scaled = gain * audio

    # Prevent clipping
    return promovits.loudness.limit(scaled)
