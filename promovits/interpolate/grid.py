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

    # Convert rates to indices
    indices = torch.full((target_frames + 2,), 0.)
    indices[1:-1] = torch.cumsum(torch.tensor(rates), 0)
    indices[-1] = source_frames - 1

    # Resample so indices are edge-aligned
    return torch.nn.functional.interpolate(
        indices[None, None],
        len(indices) - 2,
        mode='linear',
        align_corners=True)[0]
