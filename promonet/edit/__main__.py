import argparse
from pathlib import Path

import promonet


###############################################################################
# Edit
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Perform evaluation')
    parser.add_argument(
        'pitch_files',
        type=Path,
        help='The pitch files')
    parser.add_argument(
        'periodicity_files',
        type=Path,
        help='The periodicity files')
    parser.add_argument(
        'loudness_files',
        type=Path,
        help='The alignment files')
    parser.add_argument(
        'ppgs_files',
        type=Path,
        help='The ppgs files')
    parser.add_argument(
        '--output_prefixes',
        required=True,
        type=Path,
        help='The locations to save output files, minus extension.')
    parser.add_argument(
        '--pitch_shift_cents',
        type=float,
        help='Amount of pitch-shifting in cents')
    parser.add_argument(
        '--time_stretch_ratio',
        type=float,
        help='Amount of time-stretching. Makes speech faster when above one.')
    parser.add_argument(
        '--loudness_scale_db',
        type=float,
        help='Amount of loudness scaling in dB')
    return parser.parse_known_args()[0]


promonet.edit.from_files_to_files(**vars(parse_args()))
