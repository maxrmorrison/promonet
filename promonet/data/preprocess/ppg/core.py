import torch
import tqdm
from pathlib import Path

import ppgs

import promonet


###############################################################################
# Constants
###############################################################################


# PPG model checkpoint file
CHECKPOINT_FILE = promonet.ASSETS_DIR / 'checkpoints' / 'ppg.pt'

# PPG model configuration
CONFIG_FILE = promonet.ASSETS_DIR / 'configs' / 'ppg.yaml'

# Directory containing PPG checkpoints
PPG_DIR = Path.home() / 'ppgs' / 'ppgs' / 'assets'

# Sample rate of the PPG model
SAMPLE_RATE = 16000

WINDOW_SIZE = 1024
HOPSIZE = 160


###############################################################################
# Phonetic posteriorgram
###############################################################################


def from_audio(
    audio,
    sample_rate=promonet.SAMPLE_RATE,
    config=CONFIG_FILE,
    checkpoint_file=CHECKPOINT_FILE,
    gpu=None):
    """Compute PPGs from audio"""
    if promonet.PPG_MODEL is None:
        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

        # Cache model
        if not hasattr(from_audio, 'model'):
            from_audio.model = promonet.data.preprocess.ppg.conformer_ppg_model.build_ppg_model.load_ppg_model(
                config,
                checkpoint_file,
                device)

        # Maybe resample
        audio = promonet.resample(audio, sample_rate, SAMPLE_RATE)

        # Setup features
        audio = audio.to(device)
        pad = WINDOW_SIZE//2 - HOPSIZE//2
        length = torch.tensor([audio.shape[-1]], dtype=torch.long, device=device) + 2*pad #needs to be caluclated prior to padding
        audio = torch.nn.functional.pad(audio, (pad, pad))

        # Infer ppgs
        with torch.no_grad():
            return from_audio.model(audio, length)[0].T

    elif promonet.PPG_MODEL == 'senone-base':
        name = 'basemodel'
        preprocess_only = True
    elif promonet.PPG_MODEL == 'senone-phoneme':
        name = 'basemodel'
        preprocess_only = False
    elif promonet.PPG_MODEL == 'w2v2-base':
        name = 'basemodelW2V2'
        preprocess_only = True
    elif promonet.PPG_MODEL == 'w2v2-phoneme':
        name = 'basemodelW2V2'
        preprocess_only = False
    else:
        raise ValueError(f'Model {promonet.PPG_MODEL} is not defined')

    # Infer ppgs
    return ppgs.from_audio(
        audio,
        sample_rate,
        preprocess_only=preprocess_only,
        checkpoint=PPG_DIR / 'checkpoints' / f'{name}.pt',
        gpu=gpu)


def from_file(audio_file, gpu=None):
    """Compute PPGs from audio file"""
    return from_audio(promonet.load.audio(audio_file), gpu=gpu).cpu()


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute PPGs from audio file and save to disk"""
    ppg = from_file(audio_file, gpu)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute PPGs from audio files and save to disk"""
    iterator = tqdm.tqdm(
        zip(audio_files, output_files),
        desc='Extracting PPGs',
        total=len(audio_files),
        dynamic_ncols=True)
    for audio_file, output_file in iterator:
        from_file_to_file(audio_file, output_file, gpu)
