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
        device=tensor.device)[None]


def from_alignments(source, target):
    """Create time-stretch grid to convert source alignment to target"""
    # Get number of source and target frames
    source_frames = int(
        source.duration() * promovits.SAMPLE_RATE / promovits.HOPSIZE)
    target_frames = int(
        target.duration() * promovits.SAMPLE_RATE / promovits.HOPSIZE)

    # Get relative rate at each frame
    rates = pypar.compare.per_frame_rate(
        target,
        source,
        promovits.SAMPLE_RATE,
        promovits.HOPSIZE,
        target_frames)

    # Convert rates to indices and align edges
    indices = torch.cumsum(torch.tensor(rates), 0)
    indices -= indices[0].clone()
    indices *= (source_frames - 1) / indices[-1]

    return indices[None]
