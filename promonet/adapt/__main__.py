import yapecs
from pathlib import Path

import promonet


###############################################################################
# Speaker adaptation
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Perform speaker adaptation')
    parser.add_argument(
        '--name',
        required=True,
        help='The name of the speaker')
    parser.add_argument(
        '--files',
        type=Path,
        nargs='+',
        required=True,
        help='The audio files to use for adaptation')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='The model checkpoint directory')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The gpu to run adaptation on')
    return parser.parse_args()


promonet.adapt.speaker(**vars(parse_args()))
