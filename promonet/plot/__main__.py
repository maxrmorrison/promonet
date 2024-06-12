import yapecs
from pathlib import Path

import promonet


###############################################################################
# Plot speech representation
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Plot speech representation')
    parser.add_argument(
        '--audio_file',
        type=Path,
        required=True,
        help='The speech audio')
    parser.add_argument(
        '--output_file',
        type=Path,
        required=True,
        help='The file to save the output figure')
    parser.add_argument(
        '--target_file',
        type=Path,
        help='Optional corresponding ground truth to compare to')
    parser.add_argument(
        '--features',
        nargs='+',
        choices=promonet.DEFAULT_PLOT_FEATURES,
        default=promonet.DEFAULT_PLOT_FEATURES,
        help='The features to plot' )
    parser.add_argument(
        '--gpu',
        type=int,
        help='The GPU index')
    return parser.parse_args()


promonet.plot.from_file_to_file(**vars(parse_args()))
