import yapecs
from pathlib import Path

import promonet


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(
        description='Synthesize speech from features')
    parser.add_argument(
        '--loudness_files',
        type=Path,
        nargs='+',
        required=True,
        help='The loudness files')
    parser.add_argument(
        '--pitch_files',
        type=Path,
        nargs='+',
        required=True,
        help='The pitch files')
    parser.add_argument(
        '--periodicity_files',
        type=Path,
        nargs='+',
        required=True,
        help='The periodicity files')
    parser.add_argument(
        '--ppg_files',
        type=Path,
        nargs='+',
        required=True,
        help='The phonetic posteriorgram files')
    parser.add_argument(
        '--output_files',
        type=Path,
        nargs='+',
        required=True,
        help='The files to save the edited audio')
    parser.add_argument(
        '--speakers',
        type=int,
        nargs='+',
        help='The IDs of the speakers for voice conversion')
    parser.add_argument(
        '--spectral_balance_ratio',
        type=float,
        default=1.,
        help='> 1 for Alvin and the Chipmunks; < 1 for Patrick Star')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='The generator checkpoint')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The GPU index')
    return parser.parse_args()


promonet.synthesize.from_files_to_files(**vars(parse_args()))
