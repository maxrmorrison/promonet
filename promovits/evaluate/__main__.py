import argparse
from pathlib import Path

import promovits


###############################################################################
# Entry point
###############################################################################


def main():
    """Perform evaluation"""


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Perform evaluation')
    parser.add_argument(
        '--config',
        type=Path,
        default=promovits.DEFAULT_CONFIGURATION,
        help='The configuration file')
    parser.add_argument(
        '--datasets',
        required=True,
        nargs='+',
        help='The datasets to generate files for')
    parser.add_argument(
        '--gpus',
        type=int,
        help='The indices of the gpus to use for adaptation')
    return parser.parse_args()


promovits.evaluate.datasets(**vars(parse_args()))
