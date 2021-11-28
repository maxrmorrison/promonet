import argparse

import promovits


###############################################################################
# Entry point
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess a dataset')
    parser.add_argument(
        '--dataset',
        default='vctk',
        help='The name of the dataset to use')
    parser.add_argument(
        '--features',
        default=promovits.preprocess.ALL_FEATURES,
        nargs='+',
        help='The features to preprocess')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to use')
    return parser.parse_args()


if __name__ == '__main__':
    promovits.preprocess.dataset(**vars(parse_args()))
