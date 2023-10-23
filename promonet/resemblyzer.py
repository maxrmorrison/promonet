import resemblyzer
import torch
import torchaudio

import promonet


###############################################################################
# Resemblyzer speaker embedding
###############################################################################


def from_audio(audio):
    """Embed audio"""
    device, dtype = audio.device, audio.dtype

    # Preprocess
    audio = resemblyzer.preprocess_wav(
        audio.cpu().numpy().squeeze(),
        promonet.SAMPLE_RATE)

    # Embed
    embedding = model(device).embed_utterance(audio)

    # Cast
    return torch.tensor(embedding, dtype=dtype, device=device)


def from_file(file, gpu=None):
    """Embed audio on disk"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Load
    audio, _ = torchaudio.load(file)

    # Embed
    return from_audio(audio.to(device))


def from_file_to_file(input_file, output_file, gpu=None):
    """Embed audio on disk and save"""
    # Embed
    embedding = from_file(input_file, gpu).cpu()

    # Save
    torch.save(embedding, output_file)


def from_files(files, gpu=None):
    """Embed audio files on disk to a single embedding"""
    # Load
    audio = [resemblyzer.preprocess_wav(file) for file in files]

    # Embed
    device = 'cpu' if gpu is None else f'cuda:{gpu}'
    embedding = model(torch.device(device)).embed_speaker(audio)

    # Cast
    return torch.tensor(embedding, device=device)


def from_files_to_file(input_files, output_file, gpu=None):
    """Embed audio files to a single embedding and save"""
    # Embed
    embedding = from_files(input_files, gpu).cpu()

    # Save
    torch.save(embedding, output_file)


def from_files_to_files(input_files, output_files, gpu=None):
    """Embed audio files to independent embeddings and save"""
    for input_file, output_file in zip(input_files, output_files):
        from_file_to_file(input_file, output_file, gpu)


###############################################################################
# Utilities
###############################################################################


def model(device):
    """Cache model"""
    if not hasattr(model, 'model') or device != model.device:
        model.model = resemblyzer.VoiceEncoder(device=device)
        model.device = device
    return model.model
