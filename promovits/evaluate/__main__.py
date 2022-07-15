import argparse
from pathlib import Path

import promovits


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Perform evaluation')
    parser.add_argument(
        '--datasets',
        required=True,
        nargs='+',
        help='The datasets to evaluate')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=promovits.DEFAULT_CHECKPOINT,
        help='The checkpoint to use')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='The indices of the gpus to use for adaptation and evaluation')
    return parser.parse_known_args()[0]


promovits.evaluate.datasets(**vars(parse_args()))
