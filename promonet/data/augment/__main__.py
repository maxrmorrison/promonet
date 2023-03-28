import argparse

import promonet


###############################################################################
# Data augmentation
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Perform data augmentation')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=promonet.DATASETS,
        help='The name of the datasets to augment')
    return parser.parse_args()


if __name__ == '__main__':
    promonet.data.augment.datasets(**vars(parse_args()))
