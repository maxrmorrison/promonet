import yapecs

import promonet


###############################################################################
# Model export CLI
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Export torchscript model')
    return parser.parse_args()


if __name__ == '__main__':
    promonet.model.export.from_file_to_file(**vars(parse_args()))
