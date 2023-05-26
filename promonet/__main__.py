import argparse
from pathlib import Path

import promonet


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Perform speech editing')
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
        help='The audio to edit')
    parser.add_argument(
        '--output_files',
        type=Path,
        nargs='+',
        required=True,
        help='The files to save the edited audio')
    parser.add_argument(
        '--grid_files',
        type=Path,
        nargs='+',
        help='The interpolation grids for editing phoneme durations')
    parser.add_argument(
        '--target_loudness_files',
        type=Path,
        nargs='+',
        help='The loudness contours for editing loudness')
    parser.add_argument(
        '--target_pitch_files',
        type=Path,
        nargs='+',
        help='The pitch contours for shifting pitch')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=promonet.DEFAULT_CHECKPOINT,
        help='The generator checkpoint')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The GPU index')
    return parser.parse_args()


if __name__ == '__main__':
    promonet.from_files_to_files(**vars(parse_args()))
