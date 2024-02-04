import json
from pathlib import Path

import ppgs
import pypar
import torch
import torchaudio
import torchutil

import promonet


###############################################################################
# Loading utilities
###############################################################################


def audio(file):
    """Load audio from disk"""
    # Load
    audio, sample_rate = torchaudio.load(file)

    # Resample
    return torchaudio.functional.resample(
        audio,
        sample_rate,
        promonet.SAMPLE_RATE)


def features(prefix):
    """Load input features from file prefix"""
    if promonet.VITERBI_DECODE_PITCH:
        pitch_prefix = f'{prefix}-viterbi'
    else:
        pitch_prefix = prefix
    return (
        torch.load(f'{pitch_prefix}-pitch.pt'),
        torch.load(f'{pitch_prefix}-periodicity.pt'),
        torch.load(f'{prefix}-loudness.pt'),
        torch.load(f'{prefix}-ppg.pt'))


def partition(dataset):
    """Load partitions for dataset"""
    with open(promonet.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)


def pitch_distribution(dataset=promonet.TRAINING_DATASET, partition='train'):
    """Load pitch distribution"""
    if not hasattr(pitch_distribution, 'distribution'):

        # Location on disk
        key = ''
        if promonet.AUGMENT_LOUDNESS:
            key += '-loudness'
        if promonet.AUGMENT_PITCH:
            key += '-pitch'
        if promonet.VITERBI_DECODE_PITCH:
            key += '-viterbi'
        file = (
            promonet.ASSETS_DIR /
            'stats' /
            f'{dataset}-{promonet.PITCH_BINS}{key}.pt')

        try:

            # Load and cache distribution
            pitch_distribution.distribution = torch.load(file)

        except FileNotFoundError:

            # Get all voiced pitch frames
            allpitch = []
            dataset = promonet.data.Dataset(dataset, 'train')
            viterbi = '-viterbi' if promonet.VITERBI_DECODE_PITCH else ''
            for stem in torchutil.iterator(
                dataset.stems,
                'promonet.load.pitch_distribution'
            ):
                pitch = torch.load(
                    dataset.cache / f'{stem}{viterbi}-pitch.pt')
                periodicity = torch.load(
                    dataset.cache / f'{stem}{viterbi}-periodicity.pt')
                allpitch.append(
                    pitch[
                        torch.logical_and(
                            ~torch.isnan(pitch),
                            periodicity > promonet.VOICING_THRESHOLD)])

            # Sort
            pitch, _ = torch.sort(torch.cat(allpitch))

            # Bucket
            indices = torch.linspace(
                len(pitch) / promonet.PITCH_BINS,
                len(pitch) - 1,
                promonet.PITCH_BINS,
                dtype=torch.float64
            ).to(torch.long)
            pitch_distribution.distribution = pitch[indices]

            # Save
            torch.save(pitch_distribution.distribution, file)

    return pitch_distribution.distribution


def ppg(file, resample_length=None):
    """Load a PPG file, resample with a grid if need be, sparsify if needed"""
    # Load
    result = torch.load(file)

    # Maybe resample
    if resample_length is not None:
        result = promonet.edit.grid.sample(
            result,
            promonet.edit.grid.of_length(result, resample_length),
            promonet.PPG_INTERP_METHOD)

        # Preserve distribution
        return torch.softmax(torch.log(result + 1e-8), -2)

    return result


def text(file):
    """Load text file"""
    with open(file, encoding='utf-8') as file:
        return file.read()
