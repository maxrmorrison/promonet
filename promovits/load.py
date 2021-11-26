import logging

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
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    state_dict = model.state_dict()
    new_state_dict= {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logging.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    logging.info("Loaded checkpoint '{}' (iteration {})" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration
