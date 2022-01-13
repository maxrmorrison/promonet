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
    # Normalize to [0, 1]
    logfmin = torch.log2(torch.tensor(fmin))
    logfmax = torch.log2(torch.tensor(fmax))
    normalized = bins.to(torch.float) / (num_bins - 1)

    # Convert to hz
    hz = 2 ** ((normalized * (logfmax - logfmin)) + logfmin)

    # Clip to bounds
    return torch.clip(hz, fmin, fmax)


def hz_to_bins(
    hz,
    num_bins=promovits.PITCH_BINS,
    fmin=promovits.FMIN,
    fmax=promovits.FMAX):
    """Convert pitch in hz to bins"""
    # Clip to bounds
    hz = torch.clip(hz, fmin, fmax)

    # Normalize to [0, 1]
    logfmin = torch.log2(torch.tensor(fmin))
    logfmax = torch.log2(torch.tensor(fmax))
    centered = torch.log2(hz) - logfmin
    normalized = centered / (logfmax - logfmin)

    # Convert to integer bin
    return ((num_bins - 1) * normalized).to(torch.long)
