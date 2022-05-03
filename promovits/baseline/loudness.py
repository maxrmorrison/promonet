import pysodic
import torch

import promovits


def scale(audio, target_loudness):
    """Scale the audio to the target loudness"""
    loudness = pysodic.features.loudness(
        audio.to(torch.float64),
        promovits.SAMPLE_RATE,
        promovits.HOPSIZE / promovits.SAMPLE_RATE,
        promovits.WINDOW_SIZE / promovits.SAMPLE_RATE)

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
    return promovits.loudness.limit(scaled)
