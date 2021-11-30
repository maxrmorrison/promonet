import argparse
from pathlib import Path


###############################################################################
# Speaker adaptation
###############################################################################


def adapt(config, checkpoint, directory, gpus):
    """Perform speaker adaptation"""
    # TODO
    pass


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Perform speaker adaptation')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='The configuration file')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='The model checkpoint to adapt')
    parser.add_argument(
        '--directory',
        type=Path,
        required=True,
        help='A directory of 22.050 kHz wav files to use for adaptation')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='The gpus to run training on')
    return parser.parse_args()


if __name__ == '__main__':
    adapt(**vars(parse_args()))
