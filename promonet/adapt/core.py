import json
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio

import promonet


###############################################################################
# Speaker adaptation API
###############################################################################


def speaker(
    name: str,
    files: List[Path],
    checkpoint: Path = promonet.DEFAULT_CHECKPOINT,
    gpus: Optional[int] = None) -> Path:
    """Perform speaker adaptation

    Args:
        name: The name of the speaker
        files: The audio files to use for adaptation
        checkpoint: The model checkpoint
        gpus: The gpus to run adaptation on

    Returns:
        checkpoint: The file containing the trained generator checkpoint
    """
    # Make a new cache directory
    cache = promonet.CACHE_DIR / 'adapt' / name
    cache.mkdir(exist_ok=True, parents=True)

    # Preprocess audio
    for i, file in enumerate(files):

        # Convert to 22.05k
        audio = promonet.load.audio(file)

        # If audio is too quiet, increase the volume
        maximum = torch.abs(audio).max()
        if maximum < .35:
            audio *= .35 / maximum

        # Save to cache
        torchaudio.save(
            cache / f'{i:06d}-100.wav',
            audio,
            promonet.SAMPLE_RATE)

    if promonet.AUGMENT_PITCH:

        # Augment and get augmentation ratios
        ratios = promonet.data.augment.from_files_to_files(files)

        # Save augmentation ratios
        with open(promonet.AUGMENT_DIR / 'adapt' / f'{name}.json', 'w') as file:
            json.dump(ratios, file, indent=4)

    # Preprocess features
    promonet.data.preprocess.from_files_to_files(
        cache,
        cache.rglob('*.wav'),
        gpu=None if gpus is None else gpus[0])

    # Partition (all files are used for training)
    promonet.partition.dataset(name)

    # Directory to save configuration, checkpoints, and logs
    adapt_directory = promonet.RUNS_DIR / promonet.CONFIG / 'adapt' / name
    adapt_directory.mkdir(exist_ok=True, parents=True)

    # Maybe resume adaptation
    generator_path = promonet.checkpoint.latest_path(
        adapt_directory,
        'generator-*.pt')
    discriminator_path = promonet.checkpoint.latest_path(
        adapt_directory,
        'discriminator-*.pt')
    if generator_path and discriminator_path:
        checkpoint = adapt_directory

    # Perform adaptation and return generator checkpoint
    return promonet.train.run(
        name,
        checkpoint,
        adapt_directory,
        adapt_directory,
        adapt=True,
        gpus=gpus)
