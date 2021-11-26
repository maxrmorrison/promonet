import functools

import torch
import torchaudio
import torchcrepe

import promovits


###############################################################################
# Compute pitch representation
###############################################################################


def from_audio(audio, sample_rate=promovits.SAMPLE_RATE, gpu=None):
    """Preprocess pitch from audio"""
    # Target number of frames
    target_length = audio.shape[1] // promovits.HOPSIZE

    # Resample
    if sample_rate != torchcrepe.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            torchcrepe.SAMPLE_RATE)
        resampler = resampler.to(audio.device)
        audio = resampler(audio)

    # Resample hopsize
    hopsize = int(promovits.HOPSIZE * (torchcrepe.SAMPLE_RATE / sample_rate))

    # Pad
    padding = int((promovits.NUM_FFT - hopsize) // 2)
    audio = torch.nn.functional.pad(
        audio[None],
        (padding, padding),
        mode='reflect').squeeze(0)

    # Estimate pitch
    pitch, periodicity = torchcrepe.predict(
        audio,
        sample_rate=torchcrepe.SAMPLE_RATE,
        hop_length=hopsize,
        fmin=promovits.FMIN,
        fmax=promovits.FMAX,
        model='full',
        return_periodicity=True,
        batch_size=1024,
        device='cpu' if gpu is None else f'cuda:{gpu}',
        pad=False)

    # Set low energy frames to unvoiced
    periodicity = torchcrepe.threshold.Silence()(
        periodicity,
        audio,
        torchcrepe.SAMPLE_RATE,
        hop_length=hopsize,
        pad=False)

    # Potentially resize due to resampled integer hopsize
    if pitch.shape[1] != target_length:
        interp_fn = functools.partial(
            torch.nn.functional.interpolate,
            size=target_length,
            mode='linear',
            align_corners=False)
        pitch = 2 ** interp_fn(torch.log2(pitch)[None]).squeeze(0)
        periodicity = interp_fn(periodicity[None]).squeeze(0)

    return pitch, periodicity


def from_file(file, gpu=None):
    """Preprocess pitch from file"""
    return from_audio(promovits.load.audio(file), promovits.SAMPLE_RATE, gpu)


def from_file_to_file(input_file, output_prefix, gpu=None):
    """Preprocess pitch from file and save to disk"""
    pitch, periodicity = from_file(input_file, gpu)
    torch.save(pitch, f'{output_prefix}-pitch.pt')
    torch.save(periodicity, f'{output_prefix}-periodicity.pt')


###############################################################################
# Pitch utilities
###############################################################################


def log_hz_to_bins(pitch):
    """Convert pitch from log hz to bins"""
    logmin, logmax = torch.log2(promovits.FMIN), torch.log2(promovits.FMAX)
    pitch = (promovits.PITCH_BINS - 2) * (pitch - logmin) / (logmax - logmin)
    pitch = torch.clamp(pitch, 0, promovits.PITCH_BINS - 2)
    return pitch.to(torch.long)


def threshold(pitch, periodicity):
    """Voiced/unvoiced hysteresis thresholding"""
    if not hasattr(threshold, 'threshold'):
        threshold.threshold = torchcrepe.threshold.Hysteresis()
    return threshold.threshold(pitch, periodicity)
