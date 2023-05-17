import json

import pyfoal
import pypar
import torch
import torchaudio

import promonet


###############################################################################
# Loading utilities
###############################################################################


def audio(file):
    """Load audio from disk"""
    audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    return promonet.resample(audio, sample_rate)


def partition(dataset):
    """Load partitions for dataset"""
    with open(promonet.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)


def phonemes(file, interleave=False):
    """Load phonemes and interleave blanks"""
    # Load phonemes
    phonemes = torch.unique_consecutive(torch.load(file))[None]

    if not interleave:
        return phonemes

    # Interleave blanks
    interleaved = torch.full(
        (1, phonemes.shape[1] * 2 + 1),
        pyfoal.convert.phoneme_to_index(pypar.SILENCE),
        dtype=phonemes.dtype)
    interleaved[:, 1::2] = phonemes

    return interleaved


def pitch(file):
    """Load pitch from file"""
    pitch = torch.load(file)

    # Bound to range
    pitch[pitch < promonet.FMIN] = promonet.FMIN
    pitch[pitch > promonet.FMAX] = promonet.FMAX

    return pitch


def text(file):
    """Load text file"""
    with open(file) as file:
        return file.read()
