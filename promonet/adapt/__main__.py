import argparse
from pathlib import Path

import promonet


###############################################################################
# TODO - Speaker adaptation
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Perform speaker adaptation')
    parser.add_argument(
        '--config',
        type=Path,
        default=promonet.DEFAULT_CONFIGURATION,
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
        default=promonet.DEFAULT_CHECKPOINT,
        help='The checkpoint to use for adaptation')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='The gpus to run training on')
    return parser.parse_args()


if __name__ == '__main__':
    promonet.adapt.from_files_to_files(**vars(parse_args()))
