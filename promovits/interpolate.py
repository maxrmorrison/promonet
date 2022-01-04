import torch


###############################################################################
# Feature interpolation
###############################################################################


def linear(sequence, stretch):
    """Linearly interpolate sequence using an index map"""
    stretch = 2. * stretch / (sequence.shape[-1] - 1) - 1.
    return torch.nn.functional.grid_sample(
        sequence[None, None],
        torch.cat((torch.zeros_like(stretch), stretch))[None, None])[0, 0]


def pitch(sequence, stretch):
    """Interpolate pitch using an index map"""
    return 2 ** linear(torch.log2(sequence), stretch)


def ppgs(sequence, stretch):
    """Interpolate ppgs using an index map"""
    stretch = 2. * stretch / (sequence.shape[-1] - 1) - 1.
    return torch.nn.functional.grid_sample(
        sequence[None, None],
        torch.stack((torch.zeros_like(stretch), stretch))[None, None],
        mode='bilinear')[0, 0]
