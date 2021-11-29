import json

import torch
import torchaudio

import promovits


###############################################################################
# Loading utilities
###############################################################################


def audio(file):
    """Load audio from disk"""
    audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    if sample_rate != promovits.SAMPLE_RATE:
        resample_fn = torchaudio.transform.Resample(
            sample_rate,
            promovits.SAMPLE_RATE)
        audio = resample_fn(audio)

    return audio


def checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint from file"""
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    # Restore model
    model.load_state_dict(checkpoint_dict['model'])

    # Restore optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    # Restore training state
    iteration = checkpoint_dict['iteration']

    print("Loaded checkpoint '{}' (iteration {})" .format(
        checkpoint_path,
        iteration))

    return model, optimizer, iteration


def config(file):
    with open(file) as file:
        return promovits.HParams(**json.load(file))


def partition(dataset):
    """Load partitions for dataset"""
    with open(promovits.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)


def phonemes(file):
    """Load phonemes and interleave blanks"""
    phonemes = torch.load(file)
    interleaved = torch.zeros((1, len(phonemes) * 2 + 1))
    interleaved[1::2] = phonemes
    return interleaved


def text(file):
    """Load text file"""
    with open(file) as file:
        return file.read()
