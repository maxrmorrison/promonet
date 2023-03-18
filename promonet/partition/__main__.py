import argparse

import promonet


###############################################################################
# Partition datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='The datasets to partition')
    return parser.parse_args()


if __name__ == '__main__':
    promonet.partition.datasets(**vars(parse_args()))
