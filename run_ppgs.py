import argparse
from pathlib import Path

import ppgs

import promonet


###############################################################################
# Constants
###############################################################################


# TODO - remove hard coding
PPG_DIR = Path.home() / 'ppgs' / 'ppgs' / 'assets'


###############################################################################
# Preprocess various types of PPGs
###############################################################################


def main(model, gpu=None):
    # Resolve model argument
    if model == 'senone-base':
        name = 'basemodel'
        preprocess_only = True
    elif model == 'senone-phoneme':
        name = 'basemodel'
        preprocess_only = False
    elif model == 'w2v2-base':
        name = 'basemodelW2V2'
        preprocess_only = True
    elif model == 'w2v2-phoneme':
        name = 'basemodelW2V2'
        preprocess_only = False
    else:
        raise ValueError(f'Model {model} is not defined')

    # Get audio files
    audio_files = sorted(list(promonet.CACHE_DIR.rglob('*.wav')))
    audio_files = [
            file for file in audio_files if '-template' not in file.stem]

    # Get output file paths
    output_files = [
            file.parent / f'{file.stem}-ppg-{model}.pt'
            for file in audio_files]

    # Extract PPGs
    ppgs.from_files_to_files(
        audio_files,
        output_files,
        preprocess_only,
        PPG_DIR / 'checkpoints' / f'{name}.pt',
        gpu)


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
            description='Preprocess various types of PPGs')
    parser.add_argument(
        'model',
        help='The type of PPG to use')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the GPU to use for inference. Defaults to CPU.')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    main(**vars(parse_args()))
