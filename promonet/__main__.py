import yapecs
from pathlib import Path

import promonet


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Perform speech editing')
    # parser.add_argument(
    #     '--config',
    #     type=Path,
    #     default=promonet.DEFAULT_CONFIGURATION,
    #     nargs='+,'
    #     help='The configuration file')
    parser.add_argument(
        '--audio_files',
        type=Path,
        nargs='+',
        required=True,
        help='The audio to edit')
    parser.add_argument(
        '--output_files',
        type=Path,
        nargs='+',
        required=True,
        help='The files to save the edited audio')
    parser.add_argument(
        '--alignment_files',
        type=Path,
        nargs='+',
        help='The alignment files for editing phoneme durations')
    parser.add_argument(
        '--target_loudness_files',
        type=Path,
        nargs='+',
        help='The loudness contours for editing loudness')
    parser.add_argument(
        '--target_pitch_files',
        type=Path,
        nargs='+',
        help='The pitch contours for shifting pitch')
    parser.add_argument(
        '--target_ppg_files',
        type=Path,
        nargs='+',
        help='The ppgs for pronunciation editing')
    parser.add_argument(
        '--speaker_ids',
        type=int,
        nargs='+',
        help='The IDs of the speakers for voice conversion')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=promonet.DEFAULT_CHECKPOINT,
        help='The generator checkpoint')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The GPU index')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = parse_args()
    promonet.from_files_to_files(**vars(parse_args()))
