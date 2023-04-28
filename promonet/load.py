import json

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


def phonemes(file):
    """Load phonemes and interleave blanks"""
    phonemes = torch.unique_consecutive(torch.load(file))[None]
    interleaved = torch.zeros((1, phonemes.shape[1] * 2 + 1))
    interleaved[:, 1::2] = phonemes
    return interleaved


def text(file):
    """Load text file"""
    with open(file) as file:
        return file.read()
