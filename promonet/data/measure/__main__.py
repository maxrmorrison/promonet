import argparse

import promonet
import ppgs


###############################################################################
# Purge datasets
###############################################################################

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Measure dataset disk usage')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['vctk', 'daps', 'librispeech'],
        choices=['vctk', 'daps', 'librispeech'],
        help="The datasets to measure"
    )
    parser.add_argument(
        '--features',
        nargs='+',
        default=ppgs.preprocess.ALL_FEATURES,
        choices=ppgs.preprocess.ALL_FEATURES,
        help="Which cached features to measure"
    )
    parser.add_argument(
        '--unit',
        default='B',
        choices=['B', 'KB', 'MB', 'GB', 'TB'],
        help='Unit to print filesizes with'
    )
    return parser.parse_args()


promonet.data.measure.datasets(**vars(parse_args()))
