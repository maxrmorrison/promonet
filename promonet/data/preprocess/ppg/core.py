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

    # Infer ppgs
    if '-latents' in promonet.PPG_MODEL:
        latents = ppgs.preprocess.from_audio(
            audio=audio,
            sample_rate=sample_rate,
            representation=promonet.PPG_MODEL.split('-')[0],
            gpu=gpu
        )
        if ppgs.FRONTEND is not None:
            if not hasattr(from_audio, 'frontend'):
                from_audio.frontend = ppgs.FRONTEND(latents.device)
            latents = from_audio.frontend(latents)
        return latents
    return ppgs.from_audio(
        audio=audio,
        sample_rate=sample_rate,
        representation=promonet.PPG_MODEL.split('-')[0],
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
    if promonet.PPG_MODEL is None:
        iterator = tqdm.tqdm(
            zip(audio_files, output_files),
            desc='Extracting PPGs',
            total=len(audio_files),
            dynamic_ncols=True)
        for audio_file, output_file in iterator:
            from_file_to_file(audio_file, output_file, gpu)
    else:
        if '-latents' in promonet.PPG_MODEL:
            ppgs.preprocess.from_files_to_files(
                audio_files=audio_files,
                output_files=output_files,
                representation=promonet.PPG_MODEL.split('-')[0],
                num_workers=promonet.NUM_WORKERS,
                gpu=gpu
            )
        elif '-ppg' in promonet.PPG_MODEL:
            ppgs.from_files_to_files(
                audio_files=audio_files,
                output=output_files,
                representation=promonet.PPG_MODEL.split('-')[0],
                num_workers=promonet.NUM_WORKERS,
                gpu=gpu
            )
