import yapecs

import promonet


###############################################################################
# Measure datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Measure dataset disk usage')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=promonet.DATASETS,
        choices=promonet.DATASETS,
        help='The datasets to measure')
    parser.add_argument(
        '--features',
        nargs='+',
        default=promonet.ALL_FEATURES,
        choices=promonet.ALL_FEATURES,
        help='Which cached features to measure')
    return parser.parse_args()


promonet.data.measure.datasets(**vars(parse_args()))
