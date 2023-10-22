from pathlib import Path

import yapecs

import promonet


###############################################################################
# Preprocess
###############################################################################


def parse_args():
    parser = yapecs.ArgumentParser(description='Preprocess')
    parser.add_argument(
        '--files',
        nargs='+',
        type=Path,
        required=True,
        help='Audio files to preprocess')
    parser.add_argument(
        '--output_prefixes',
        nargs='+',
        type=Path,
        help='Files to save features, minus extension')
    parser.add_argument(
        '--features',
        default=promonet.INPUT_FEATURES,
        choices=promonet.INPUT_FEATURES,
        nargs='+',
        help='The features to preprocess')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to use')
    return parser.parse_args()


promonet.preprocess.from_files_to_files(**vars(parse_args()))
