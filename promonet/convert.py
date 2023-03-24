import warnings

import torch

import promonet


###############################################################################
# Unit conversions
###############################################################################


def bins_to_hz(
    bins,
    num_bins=promonet.PITCH_BINS,
    fmin=promonet.FMIN,
    fmax=promonet.FMAX):
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
    num_bins=promonet.PITCH_BINS,
    fmin=promonet.FMIN,
    fmax=promonet.FMAX):
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


###############################################################################
# Time conversions
###############################################################################


def seconds_to_frames(seconds):
    """Convert seconds to frames"""
    return int(seconds * promonet.SAMPLE_RATE / promonet.HOPSIZE)


def frames_to_samples(frames):
    """Convert number of frames to samples"""
    return frames * promonet.HOPSIZE


def frames_to_seconds(frames):
    """Convert number of frames to seconds"""
    return frames * samples_to_seconds(promonet.HOPSIZE)


def samples_to_seconds(samples, sample_rate=promonet.SAMPLE_RATE):
    """Convert time in samples to seconds"""
    return samples / sample_rate


def samples_to_frames(samples):
    """Convert time in samples to frames"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return samples // promonet.HOPSIZE
    