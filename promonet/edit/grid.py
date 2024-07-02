import ppgs
import torch

import promonet


###############################################################################
# Grid sampling
###############################################################################


def sample(sequence, grid, method='linear'):
    """Perform 1D grid-based sampling"""
    # Linear grid interpolation
    if method == 'linear':
        x = grid
        fp = sequence

        # Input indices
        xp = torch.arange(fp.shape[-1], device=fp.device)

        # Output indices
        i = torch.searchsorted(xp, x, side='right')

        # Replicate final frame
        # "replication_pad1d_cpu" not implemented for 'Half'
        if fp.dtype == torch.float16:
            fp = torch.nn.functional.pad(
                fp.to(torch.float32),
                (0, 1),
                mode='replicate'
            ).to(torch.float16)
        else:
            fp = torch.nn.functional.pad(fp, (0, 1), mode='replicate')
        xp = torch.cat((xp, xp[-1:] + 1))

        # Interpolate
        return fp[..., i - 1] * (xp[i] - x) + fp[..., i] * (x - xp[i - 1])

    # Nearest neighbors grid interpolation
    elif method == 'nearest':
        return sequence[..., torch.round(grid).to(torch.long)]

    else:
        raise ValueError(f'Grid sampling method {method} is not defined')


###############################################################################
# Interpolation grids
###############################################################################


def constant(tensor, ratio):
    """Create a grid for constant-ratio time-stretching"""
    return ppgs.edit.grid.constant(tensor, ratio)


def from_alignments(source, target):
    """Create time-stretch grid to convert source alignment to target"""
    return ppgs.edit.grid.from_alignments(
        source,
        target,
        sample_rate=promonet.SAMPLE_RATE,
        hopsize=promonet.HOPSIZE)


def of_length(tensor, length):
    """Create time-stretch grid of a specified length"""
    return ppgs.edit.grid.of_length(tensor, length)
