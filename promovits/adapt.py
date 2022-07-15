import argparse
import shutil
from pathlib import Path

import torch
import torchaudio

import promovits


###############################################################################
# Speaker adaptation
###############################################################################


def speaker(
    config,
    name,
    directory,
    checkpoint=promovits.DEFAULT_CHECKPOINT,
    gpus=None):
    """Perform speaker adaptation"""
    cache = promovits.CACHE_DIR / name
    cache.mkdir(exist_ok=True, parents=True)

    # Preprocess
    with promovits.data.chdir(cache):

        # Preprocess audio
        for i, file in enumerate(directory.rglob('.wav')):

            # Convert to 22.05k
            audio = promovits.load.audio(file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # Save to cache
            torchaudio.save(f'{i:06d}.wav', audio, promovits.SAMPLE_RATE)

        # Preprocess features
        promovits.preprocess.files(
            Path(),
            Path().rglob('*.wav'),
            features=['ppg', 'prosody', 'spectrogram'],
            gpu=None if gpus is None else gpus[0])

    # Partition (all files are used for training)
    promovits.partition.dataset(name)

    # Directory to save configuration, checkpoints, and logs
    adapt_directory = promovits.RUNS_DIR / config.stem / 'adapt' / name
    adapt_directory.mkdir(exist_ok=True, parents=True)

    # Save configuration
    shutil.copyfile(config, adapt_directory / config.name)

    # Perform adaptation and return generator checkpoint
    return promovits.train.run(
        name,
        checkpoint,
        adapt_directory,
        adapt_directory,
        adapt=True,
        gpus=gpus)


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Perform speaker adaptation')
    parser.add_argument(
        '--config',
        type=Path,
        default=promovits.DEFAULT_CONFIGURATION,
        help='The configuration file')
    parser.add_argument(
        '--name',
        required=True,
        help='The name of the speaker')
    parser.add_argument(
        '--directory',
        type=Path,
        required=True,
        help='Directory containing speech audio for speaker adaptation')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=promovits.DEFAULT_CHECKPOINT,
        help='The checkpoint to use for adaptation')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='The gpus to run training on')
    return parser.parse_args()


if __name__ == '__main__':
    speaker(**vars(parse_args()))
