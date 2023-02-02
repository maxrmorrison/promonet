import argparse

import promonet


###############################################################################
# Entry point
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess a dataset')
    parser.add_argument(
        '--config',
        nargs='+',
        help='The name of the datasets to use')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['daps', 'vctk'],
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

    # Delete config files
    args = parser.parse_args()
    del args.config
    return args



if __name__ == '__main__':
    promonet.preprocess.datasets(**vars(parse_args()))
