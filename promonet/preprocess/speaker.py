import torch
import torchaudio
import torchutil
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

import promonet


###############################################################################
# Constants
###############################################################################


# Maximum batch size for batched WavLM inference
WAVLM_MAX_BATCH_SIZE = 16

# Sample rate of the WavLM model audio input
WAVLM_SAMPLE_RATE = 16000


###############################################################################
# WavLM x-vector speaker embedding
###############################################################################


def from_audio(audio, sample_rate=promonet.SAMPLE_RATE, gpu=None):
    """Compute speaker embedding from audio"""
    # Resample
    torchaudio.functional.resample(audio, sample_rate, WAVLM_SAMPLE_RATE)

    # Embed
    return infer(audio[0], gpu)


def from_file(file, gpu=None):
    """Compute speaker embedding from file"""
    return from_audio(promonet.load.audio(file), gpu=gpu)


def from_file_to_file(file, output_file, gpu=None):
    """Compute speaker embedding from file and save"""
    # Embed
    embedding = from_file(file, gpu).cpu()

    # Save
    torch.save(embedding, output_file)


def from_files_to_files(files, output_files, gpu=None):
    """Compute speaker embedding from files and save"""
    for file, output_file in torchutil.iterator(
        zip(files, output_files),
        'WavLM x-vectors',
        total=len(files)
    ):
        from_file_to_file(file, output_file, gpu)


###############################################################################
# Utilities
###############################################################################


def infer(audio, gpu=None):
    """Infer speaker embedding from audio"""
    # Cache networks
    if not hasattr(infer, 'feature_extractor'):
        infer.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            'microsoft/wavlm-base-plus-sv')
    if not hasattr(infer, 'model'):
        infer.model = WavLMForXVector.from_pretrained(
            'microsoft/wavlm-base-plus-sv')

    # Place on device (no-op if devices match)
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    infer.model.to(device)

    # Preprocess
    features = infer.feature_extractor(
        audio,
        padding=True,
        return_tensors="pt")

    # Embed
    embeddings = infer.model(
        features['input_values'].to(device),
        features['attention_mask'].to(device)
    ).embeddings.detach()

    # Normalize
    return torch.nn.functional.normalize(embeddings, dim=-1)
