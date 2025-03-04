import yapecs

import promonet
from pathlib import Path

###############################################################################
# Data augmentation
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Perform data augmentation')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=promonet.DATASETS,
        help='The name of the datasets to augment')
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


promonet.data.augment.datasets(**vars(parse_args()))
