import contextlib

import torch
import torchaudio

import promonet


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def generation_context(model):
    device_type = next(model.parameters()).device.type

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation; turn on mixed precision
    with torch.inference_mode(), torch.autocast(device_type):
        yield

    # Prepare model for training
    model.train()


def resample(audio, sample_rate, target_rate=promonet.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)
