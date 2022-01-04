import torch

import promovits


###############################################################################
# Feature interpolation
###############################################################################


def features(sequence, stretch):
    """Interpolate features using an index map"""
    index = promovits.PPG_CHANNELS

    # Interpolate PPGs
    feats = ppgs(sequence[0, :index], stretch)

    # Maybe interpolate loudness
    if promovits.LOUDNESS_FEATURES:
        feats = torch.cat((
            feats,
            linear(sequence[0, index:index + 1], stretch)))

    # Maybe interpolate periodicity
    if promovits.PERIODICITY_FEATURES:
        feats = torch.cat((feats, linear(sequence[0, -1:], stretch)))

    return feats


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
    if promovits.PPG_INTERP_METHOD == 'linear':
        mode = 'bilinear'
    elif promovits.PPG_INTERP_METHOD == 'nearest':
        mode = 'nearest'
    else:
        raise ValueError(
            f'Interpolation mode {promovits.PPG_INTERP_METHOD} is not defined')

    stretch = 2. * stretch / (sequence.shape[-1] - 1) - 1.
    return torch.nn.functional.grid_sample(
        sequence[None, None],
        torch.stack((torch.zeros_like(stretch), stretch))[None, None],
        mode)[0, 0]


###############################################################################
# Interpolation utilities
###############################################################################


def constant_stretch(length, ratio):
    """Create an index map for a constant-ratio time-stretch"""
    return torch.linspace(0., length, int(length / ratio), dtype=torch.float)
