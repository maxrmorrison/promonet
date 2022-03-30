import torch

import pypar

import promovits


###############################################################################
# Interpolation grids
###############################################################################


def constant(tensor, ratio):
    """Create a grid for constant-ratio time-stretching"""
    return torch.linspace(
        0.,
        tensor.shape[-1] - 1,
        round((tensor.shape[-1]) / ratio + 1e-4),
        dtype=torch.float,
        device=tensor.device)


def from_alignments(source, target):
    """Create time-stretch grid to convert source alignment to target"""
    # Get relative rate at each frame
    rates = pypar.compare.per_frame_rate(
        target,
        source,
        promovits.SAMPLE_RATE,
        promovits.HOPSIZE)

    # Convert rates to indices
    frames = int(source.duration() * promovits.SAMPLE_RATE / promovits.HOPSIZE)
    indices = torch.full(frames, 0.)
    indices[1:] = torch.cumsum(rates)
    indices[-1] = frames

    return indices
