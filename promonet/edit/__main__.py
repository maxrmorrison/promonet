import yapecs
from pathlib import Path

import promonet


###############################################################################
# Edit
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Edit speech representation')
    parser.add_argument(
        '--loudness_files',
        type=Path,
        nargs='+',
        required=True,
        help='The loudness files to edit')
    parser.add_argument(
        '--pitch_files',
        type=Path,
        nargs='+',
        required=True,
        help='The pitch files to edit')
    parser.add_argument(
        '--periodicity_files',
        type=Path,
        nargs='+',
        required=True,
        help='The periodicity files to edit')
    parser.add_argument(
        '--ppg_files',
        type=Path,
        nargs='+',
        required=True,
        help='The ppg files to edit')
    parser.add_argument(
        '--output_prefixes',
        required=True,
        type=Path,
        nargs='+',
        help='The locations to save output files, minus extension')
    parser.add_argument(
        '--pitch_shift_cents',
        type=float,
        help='Amount of pitch-shifting in cents')
    parser.add_argument(
        '--time_stretch_ratio',
        type=float,
        help='Amount of time-stretching. Faster when above one.')
    parser.add_argument(
        '--loudness_scale_db',
        type=float,
        help='Amount of loudness scaling in dB')
    parser.add_argument(
        '--stretch_unvoiced',
        action='store_true',
        help='If provided, applies time-stretching to unvoiced frames')
    parser.add_argument(
        '--stretch_silence',
        action='store_true',
        help='If provided, applies time-stretching to silence frames')
    parser.add_argument(
        '--save_grid',
        action='store_true',
        help='If provided, also saves the time-stretch grid')
    return parser.parse_args()


promonet.edit.from_files_to_files(**vars(parse_args()))
