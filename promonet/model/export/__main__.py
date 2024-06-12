from pathlib import Path

import yapecs

import promonet


###############################################################################
# Model export CLI
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Export torchscript model')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='The generator checkpoint')
    parser.add_argument(
        '--output_file',
        type=Path,
        default='promonet-export.ts',
        help='The torch file to write the exported model')

    return parser.parse_args()


if __name__ == '__main__':
    promonet.model.export.from_file_to_file(**vars(parse_args()))
