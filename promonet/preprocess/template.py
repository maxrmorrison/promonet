import functools
import math
import multiprocessing as mp

import torch
import torchaudio
import tqdm

import promonet


###############################################################################
# Constants
###############################################################################


# Computed billions of times, so worthwhile to cache
SAMPLE_RATE_RADIANS = 2 * math.pi / promonet.SAMPLE_RATE


###############################################################################
# Templating
###############################################################################


def from_prosody(pitch, periodicity, loudness):
    """Compute waveform template from features"""
    # Interpolate prosody
    interp_fn = functools.partial(
        torch.nn.functional.interpolate,
        scale_factor=promonet.HOPSIZE,
        mode='linear',
        align_corners=False)
    pitch = 2 ** interp_fn(torch.log2(pitch)[:, None]).squeeze()
    periodicity = interp_fn(periodicity[:, None]).squeeze()

    # Create template
    result = torch.zeros_like(pitch)
    for i in range(1, len(result)):

        # Autoregressively generate unwrapped phase
        result[i] = result[i - 1] + SAMPLE_RATE_RADIANS * pitch[i]

    # Convert phase to pitched waveform
    result = torch.cos(result)[None]
    # result = sawtooth(result)[None]

    # Add some noise
    result += periodicity * (2 * torch.rand_like(result) - 1)

    # Apply loudness scaling
    return promonet.baseline.loudness.scale(result, loudness)


def from_file_to_file(prefix):
    """Compute waveform template from cached files"""
    pitch = promonet.load.pitch(f'{prefix}-pitch.pt')
    periodicity = torch.load(f'{prefix}-periodicity.pt')
    loudness = torch.load(f'{prefix}-loudness.pt')
    template = from_prosody(pitch, periodicity, loudness)
    torchaudio.save(f'{prefix}-template.wav', template, promonet.SAMPLE_RATE)


def from_files_to_files(prefixes):
    """Compute waveform templates from cached files"""
    with mp.get_context('spawn').Pool() as pool:
        pool.map(from_file_to_file, prefixes)
    # iterator = tqdm.tqdm(prefixes)
    # for prefix in prefixes:
    #     from_file_to_file(prefix)
    #     print(prefix)


###############################################################################
# Utilities
###############################################################################


def sawtooth(phase):
    """Compute sawtooth wave from unwrapped values"""
    result = torch.zeros_like(phase)

    # Wrap phase
    phase %= 2 * math.pi

    # Map phase to sawtooth values
    quadrants = [
        (phase < math.pi / 2).to(torch.bool),
        (math.pi / 2 < phase).to(torch.bool) &
            (phase < math.pi).to(torch.bool),
        (math.pi < phase).to(torch.bool) &
            (phase < 3 * math.pi / 2).to(torch.bool),
        (3 * math.pi / 2 < phase).to(torch.bool) &
            (phase < 2 * math.pi).to(torch.bool)]

    result[quadrants[0]] = 2 * result[quadrants[0]] / math.pi
    result[quadrants[1]] = 1 - 2 * (result[quadrants[1]] - math.pi / 2) / math.pi
    result[quadrants[2]] = - 2 * (result[quadrants[2]] - math.pi) / math.pi
    result[quadrants[3]] = -1 + 2 * (result[quadrants[3]] - 3 * math.pi / 2) / math.pi

    return result
