import argparse

import promonet


###############################################################################
# Entry point
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess a dataset')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['vctk'],
        help='The name of the datasets to use')
    parser.add_argument(
        '--features',
        default=promonet.preprocess.ALL_FEATURES,
        nargs='+',
        help='The features to preprocess')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to use')
    return parser.parse_args()


if __name__ == '__main__':
    promonet.preprocess.datasets(**vars(parse_args()))
