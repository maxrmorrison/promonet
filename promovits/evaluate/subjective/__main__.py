import argparse
from pathlib import Path

import promovits


###############################################################################
# Constants
###############################################################################


NUM_SAMPLES = 50


###############################################################################
# Generate files for subjective evaluation
###############################################################################


def main(config_file, datasets, checkpoint, gpu=None):
    """Generate files"""
    # Load config
    config = promovits.load.config(config_file)

    # Generate files for each dataset
    for dataset in datasets:

        # Get dataset
        dataset = promovits.data.dataset(dataset, 'test')

        # TODO - Generate files for dataset
        pass


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate files for subjective evaluation')
    parser.add_argument(
        '--config_file',
        type=Path,
        required=True,
        help='The configuration file')
    parser.add_argument(
        '--datasets',
        required=True,
        nargs='+',
        help='The datasets to generate files for')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='The model checkpoint to use for evaluation')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to use')
    return parser.parse_args()


main(**vars(parse_args()))
