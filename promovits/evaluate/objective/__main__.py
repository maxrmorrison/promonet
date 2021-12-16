import argparse
from pathlib import Path

import promovits


###############################################################################
# Constants
###############################################################################


NUM_SAMPLES = 128


###############################################################################
# Perform objective evaluation
###############################################################################


def main(config, datasets, gpu=None):
    """Generate files"""
    name = config.stem

    # Generate files for each dataset
    for dataset in datasets:

        # TODO - Generate files for dataset
        pass


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Perform objective evaluation')
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
        '--gpu',
        type=int,
        help='The index of the gpu to use')
    return parser.parse_args()


main(**vars(parse_args()))
