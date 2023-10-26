import yapecs

import promonet


###############################################################################
# Download datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Download datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=promonet.DATASETS,
        default=promonet.DATASETS,
        help='The datasets to download')
    return parser.parse_args()


promonet.data.download.datasets(**vars(parse_args()))
