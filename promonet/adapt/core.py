from pathlib import Path
from typing import List, Optional

import huggingface_hub
import torch
import torchaudio
import torchutil

import promonet


###############################################################################
# Speaker adaptation API
###############################################################################


def speaker(
    name: str,
    files: List[Path],
    checkpoint: Optional[Path] = None,
    gpu: Optional[int] = None
) -> Path:
    """Perform speaker adaptation

    Args:
        name: The name of the speaker
        files: The audio files to use for adaptation
        checkpoint: The model checkpoint directory
        gpu: The gpu to run adaptation on

    Returns:
        checkpoint: The file containing the trained generator checkpoint
    """
    # Make a new cache directory
    cache = promonet.CACHE_DIR / 'adapt' / name
    cache.mkdir(exist_ok=True, parents=True)

    # Preprocess audio
    output_files = []
    for i, file in enumerate(files):

        # Convert to 22.05k
        audio = promonet.load.audio(file)

        # If audio is too quiet, increase the volume
        maximum = torch.abs(audio).max()
        if maximum < .35:
            audio *= .35 / maximum

        # Save to cache
        output_file = cache / f'{i:06d}-100.wav'
        output_files.append(output_file)
        torchaudio.save(
            cache / f'{i:06d}-100.wav',
            audio,
            promonet.SAMPLE_RATE)

    # from this point on we want to deal with the files in cache, not the original files
    files = output_files

    promonet.preprocess.from_files_to_files(
        files,
        output_prefixes=[file.parent / file.stem.split("-")[0] for file in files],
        gpu=gpu,
        features=[
            'text',
        ])

    if promonet.AUGMENT_PITCH or promonet.AUGMENT_LOUDNESS:

        # Augment and get augmentation ratios
        promonet.data.augment.from_files_to_files(files, name)

    # Preprocess features
    promonet.preprocess.from_files_to_files(
        list(cache.rglob('*.wav')),
        gpu=gpu,
        features=[
            'loudness',
            'pitch',
            'periodicity',
            'ppg',
            'spectrogram'
        ])

    # Partition (all files are used for training)
    promonet.partition.datasets([name])

    # Directory to save configuration, checkpoints, and logs
    directory = promonet.RUNS_DIR / promonet.CONFIG / 'adapt' / name
    directory.mkdir(exist_ok=True, parents=True)

    # Maybe resume adaptation
    generator_path = torchutil.checkpoint.latest_path(
        directory,
        'generator-*.pt')
    discriminator_path = torchutil.checkpoint.latest_path(
        directory,
        'discriminator-*.pt')
    if generator_path and discriminator_path:
        checkpoint = directory

    # Maybe download checkpoint
    if checkpoint is None:
        generator_checkpoint = huggingface_hub.hf_hub_download(
            'maxrmorrison/promonet',
            f'generator-00{promonet.STEPS}.pt')
        huggingface_hub.hf_hub_download(
            'maxrmorrison/promonet',
            f'discriminator-00{promonet.STEPS}.pt')
        checkpoint = Path(generator_checkpoint).parent

    # Perform adaptation and return generator checkpoint
    return promonet.train(
        directory,
        name,
        adapt_from=checkpoint,
        gpu=gpu)
