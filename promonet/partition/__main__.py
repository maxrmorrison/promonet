import yapecs

import promonet


###############################################################################
# Partition datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        default=promonet.DATASETS,
        nargs='+',
        help='The datasets to partition')
    return parser.parse_args()


promonet.partition.datasets(**vars(parse_args()))
