import math
import warnings

import torch

import promonet


###############################################################################
# Loudness conversions
###############################################################################


def db_to_ratio(db):
    """Convert decibels to perceptual loudness ratio"""
    return 2 ** (db / 10)


def ratio_to_db(ratio):
    """Convert perceptual loudness ratio to decibels"""
    if isinstance(ratio, torch.Tensor):
        return 10 * torch.log2(ratio)
    else:
        return 10 * math.log2(ratio)


###############################################################################
# Pitch conversions
###############################################################################


def bins_to_hz(
    bins,
    num_bins=promonet.PITCH_BINS,
    fmin=promonet.FMIN,
    fmax=promonet.FMAX):
    """Convert pitch in bin indices to hz"""
    if promonet.VARIABLE_PITCH_BINS:
        # Get bin boundaries
        distribution = torch.cat([
            promonet.load.pitch_distribution(),
            torch.tensor([promonet.FMAX])])

        # Compute offset in Hz
        offset = 2 ** (
            (
                torch.log2(distribution[bins + 1]) -
                torch.log2(distribution[bins])
            ) / 2)
        return distribution[bins] + offset

    # Normalize to [0, 1]
    logfmin = torch.log2(torch.tensor(fmin))
    logfmax = torch.log2(torch.tensor(fmax))
    normalized = bins.to(torch.float) / (num_bins - 1)

    # Convert to hz
    hz = 2 ** ((normalized * (logfmax - logfmin)) + logfmin)

    # Clip to bounds
    return torch.clip(hz, fmin, fmax)


def cents_to_ratio(cents):
    """Convert pitch ratio in cents to linear ratio"""
    return 2 ** (cents / 1200)


def hz_to_bins(
    hz,
    num_bins=promonet.PITCH_BINS,
    fmin=promonet.FMIN,
    fmax=promonet.FMAX):
    """Convert pitch in hz to bins"""
    # Clip to bounds
    hz = torch.clip(hz, fmin, fmax)

    # Maybe size bins according to count
    if promonet.VARIABLE_PITCH_BINS:
        distribution = promonet.load.pitch_distribution().to(hz.device)
        bins = torch.searchsorted(distribution, hz)
        return torch.clip(bins, 0, num_bins.item() - 1)

    # Normalize to [0, 1]
    logfmin = torch.log2(fmin)
    logfmax = torch.log2(fmax)
    centered = torch.log2(hz) - logfmin
    normalized = centered / (logfmax - logfmin)

    # Convert to integer bin
    return ((num_bins - 1) * normalized).to(torch.long)


def ratio_to_cents(ratio):
    """Convert linear pitch ratio to cents"""
    return 1200 * math.log2(ratio)


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
