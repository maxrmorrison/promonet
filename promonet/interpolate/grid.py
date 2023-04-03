import torch

import pypar

import promonet


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
    # Get number of source and target frames
    source_frames = promonet.convert.seconds_to_frames(source.duration())
    target_frames = promonet.convert.seconds_to_frames(target.duration())

    # Get relative rate at each frame
    rates = pypar.compare.per_frame_rate(
        target,
        source,
        promonet.SAMPLE_RATE,
        promonet.HOPSIZE,
        target_frames)

    # Convert rates to indices and align edges
    indices = torch.cumsum(torch.tensor(rates), 0)
    indices -= indices[0].clone()
    indices *= (source_frames - 1) / indices[-1]

    return indices
