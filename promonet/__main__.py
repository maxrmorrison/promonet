import argparse
from pathlib import Path

import promonet


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Perform prosody editing')
    parser.add_argument(
        '--config',
        type=Path,
        default=promonet.DEFAULT_CONFIGURATION,
        help='The configuration file')
    parser.add_argument(
        '--audio_files',
        type=Path,
        nargs='+',
        required=True,
        help='The audio files to process')
    parser.add_argument(
        '--output_files',
        type=Path,
        nargs='+',
        required=True,
        help='The files to save the output audio')
    parser.add_argument(
        '--target_alignment_files',
        type=Path,
        nargs='+',
        help='The files with the target phoneme alignment')
    parser.add_argument(
        '--target_loudness_files',
        type=Path,
        nargs='+',
        help='The files with the per-frame target loudness')
    parser.add_argument(
        '--target_periodicity_files',
        type=Path,
        nargs='+',
        help='The files with the per-frame target periodicity')
    parser.add_argument(
        '--target_pitch_files',
        type=Path,
        nargs='+',
        help='The files with the per-frame target pitch')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=promonet.DEFAULT_CHECKPOINT,
        help='The generator checkpoint')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to use for generation')
    return parser.parse_args()


if __name__ == '__main__':
    promonet.from_files_to_files(**vars(parse_args()))