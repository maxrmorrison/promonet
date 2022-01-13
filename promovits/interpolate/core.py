import torch

import promovits


###############################################################################
# Feature interpolation
###############################################################################


def features(sequence, grid):
    """Interpolate features using a grid"""
    index = promovits.PPG_CHANNELS

    # Interpolate PPGs
    feats = ppgs(sequence[:, :index], grid)

    # Maybe interpolate loudness
    if promovits.LOUDNESS_FEATURES:
        loudness = grid_sample(sequence[:, index:index + 1], grid)
        feats = torch.cat((feats, loudness), dim=1)

    # Maybe interpolate periodicity
    if promovits.PERIODICITY_FEATURES:
        periodicity = grid_sample(sequence[:, -1:], grid)
        feats = torch.cat((feats, periodicity), dim=1)

    return feats


def pitch(sequence, grid):
    """Interpolate pitch using a grid"""
    return 2 ** grid_sample(torch.log2(sequence), grid)


def ppgs(sequence, grid):
    """Interpolate ppgs using a grid"""
    return grid_sample(sequence, grid, promovits.PPG_INTERP_METHOD)


###############################################################################
# Grid sampling
###############################################################################


def grid_sample(sequence, grid, method='linear'):
    """Perform grid-based sampling"""
    # Require interpolation method to be defined
    if method not in ['linear', 'nearest']:
        raise ValueError(
            f'Interpolation mode {promovits.PPG_INTERP_METHOD} is not defined')

    # Setup grid parameters
    x = grid
    fp = sequence

    # Linear grid interpolation
    if method == 'linear':
        xp = torch.arange(sequence.shape[-1], device=sequence.device)
        i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
        return (
            (fp[..., i - 1] * (xp[i] - x) + fp[..., i] * (x - xp[i - 1])) /
            (xp[i] - xp[i - 1]))

    # Nearest neighbors grid interpolation
    return fp[..., torch.round(x).to(torch.long)]
