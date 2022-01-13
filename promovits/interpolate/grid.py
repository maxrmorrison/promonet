import torch


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


def voiced(tensor, ratio):
    """Create a grid for time-stretching of only voiced phonemes"""
    # TODO
    pass


def vowels(tensor, ratio):
    """Create a grid for time-stretching of only vowels"""
    # TODO
    pass
