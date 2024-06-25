from pathlib import Path

import yapecs

import promonet


###############################################################################
# Pack features
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(
        description='Pack features in a single tensor')
    parser.add_argument(
        '--audio_file',
        type=Path,
        help='The audio file to convert to a packed feature tensor')
    parser.add_argument(
        '--output_file',
        type=Path,
        help=(
            'File to save packed tensor. '
            'Default is audio_file with .pt extension'))
    parser.add_argument(
        '--speaker',
        type=int,
        default=0,
        help='The speaker index')
    parser.add_argument(
        '--spectral_balance_ratio',
        type=float,
        default=1.,
        help='> 1 for Alvin and the Chipmunks; < 1 for Patrick Star')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The GPU index')
    return parser.parse_args()


promonet.data.pack.from_file_to_file(**vars(parse_args()))
