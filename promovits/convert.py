import torch

import promovits


###############################################################################
# Unit conversions
###############################################################################


def bins_to_hz(
    bins,
    num_bins=promovits.PITCH_BINS,
    fmin=promovits.FMIN,
    fmax=promovits.FMAX):
    """Convert pitch in bin indices to hz"""
    logfmin = torch.log2(torch.tensor(fmin))
    logfmax = torch.log2(torch.tensor(fmax))
    normalized = bins.to(torch.float) / (num_bins - 1)
    centered = normalized * (logfmax - logfmin)
    return 2 ** (centered + logfmin)


def hz_to_bins(
    hz,
    num_bins=promovits.PITCH_BINS,
    fmin=promovits.FMIN,
    fmax=promovits.FMAX):
    """Convert pitch in hz to bins"""
    logfmin = torch.log2(torch.tensor(fmin))
    logfmax = torch.log2(torch.tensor(fmax))
    centered = torch.log2(hz) - logfmin
    normalized = centered / (logfmax - logfmin)
    return ((num_bins - 1) * normalized).to(torch.long)
