import ppgs

import promonet


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
