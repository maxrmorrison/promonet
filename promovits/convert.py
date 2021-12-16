import torch

import promovits


###############################################################################
# Unit conversions
###############################################################################


def hz_to_bins(
    hz,
    bins=promovits.PITCH_BINS,
    fmin=promovits.FMIN,
    fmax=promovits.FMAX):
    """Convert pitch in hz to bins"""
    logfmin = torch.log2(torch.tensor(fmin))
    logfmax = torch.log2(torch.tensor(fmax))
    centered = torch.log2(hz) - logfmin
    normalized = centered / (logfmax - logfmin)
    return ((bins - 1) * normalized).to(torch.long)
