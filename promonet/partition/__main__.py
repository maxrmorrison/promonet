import yapecs

import promonet
from pathlib import Path

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
    parser.add_argument(
        '--cache_dir',
        default=promonet.CACHE_DIR,
        type=Path,
        help='Cache directory')
    parser.add_argument(
        '--assets_dir',
        default=promonet.ASSETS_DIR,
        type=Path,
        help='Assets directory')
    return parser.parse_args()


promonet.partition.datasets(**vars(parse_args()))
