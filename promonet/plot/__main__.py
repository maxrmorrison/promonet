import argparse
from pathlib import Path

import promonet


###############################################################################
# Plot prosody
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Plot pitch and alignment')
    parser.add_argument(
        '--text_file',
        type=Path,
        required=True,
        help='The speech transcript')
    parser.add_argument(
        '--audio_file',
        type=Path,
        required=True,
        help='The corresponding speech audio')
    parser.add_argument(
        '--output_file',
        type=Path,
        required=True,
        help='The file to save the output figure')
    return parser.parse_args()


if __name__ == '__main__':
    promonet.plot.from_file_to_file(**vars(parse_args()))
