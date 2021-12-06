import functools

import torch
import torchaudio
import torchcrepe

import promovits


###############################################################################
# Constants
###############################################################################


SILENCE_THRESHOLD = -60.  # dB


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

    # Compute loudness
    loudness = torchcrepe.loudness.a_weighted(
        audio,
        sample_rate,
        hopsize,
        False)

    # Set low energy frames to unvoiced
    periodicity[loudness < SILENCE_THRESHOLD] = 0.

    # Potentially resize due to resampled integer hopsize
    if pitch.shape[1] != target_length:
        interp_fn = functools.partial(
            torch.nn.functional.interpolate,
            size=target_length,
            mode='linear',
            align_corners=False)
        pitch = 2 ** interp_fn(torch.log2(pitch)[None]).squeeze(0)
        periodicity = interp_fn(periodicity[None]).squeeze(0)
        # TODO - is this the correct interpolation?
        loudness = interp_fn(loudness[None]).squeeze(0)

    return pitch, periodicity, loudness


def from_file(file, gpu=None):
    """Preprocess pitch from file"""
    return from_audio(promovits.load.audio(file), promovits.SAMPLE_RATE, gpu)


def from_file_to_file(input_file, output_prefix, gpu=None):
    """Preprocess pitch from file and save to disk"""
    pitch, periodicity, loudness = from_file(input_file, gpu)
    torch.save(pitch, f'{output_prefix}-pitch.pt')
    torch.save(periodicity, f'{output_prefix}-periodicity.pt')
    torch.save(loudness, f'{output_prefix}-loudness.pt')


def from_files_to_files(input_files, output_prefixes, gpu=None):
    """Preprocess pitch from files and save to disk"""
    for input_file, output_prefix in zip(input_files, output_prefixes):
        from_file_to_file(input_file, output_prefix, gpu)


###############################################################################
# Pitch utilities
###############################################################################


def threshold(pitch, periodicity):
    """Voiced/unvoiced hysteresis thresholding"""
    if not hasattr(threshold, 'threshold'):
        threshold.threshold = torchcrepe.threshold.Hysteresis()
    return threshold.threshold(pitch, periodicity)
