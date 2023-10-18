import argparse
import shutil
from pathlib import Path

import promonet


###############################################################################
# Entry point
###############################################################################


def main(
    config,
    dataset,
    train_partition='train',
    valid_partition='valid',
    adapt_from=False,
    gpu=None):
    # Create output directory
    directory = promonet.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    promonet.train(
        dataset,
        directory,
        train_partition,
        valid_partition,
        adapt_from,
        gpu)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--config',
        type=Path,
        nargs='+',
        default=[promonet.DEFAULT_CONFIGURATION],
        help='The configuration file')
    parser.add_argument(
        '--dataset',
        default=promonet.TRAINING_DATASET,
        help='The dataset to train on')
    parser.add_argument(
        '--train_partition',
        default='train',
        help='The data partition to train on')
    parser.add_argument(
        '--valid_partition',
        default='valid',
        help='The data partition to perform validation on')
    parser.add_argument(
        '--adapt_from',
        type=Path,
        help='A checkpoint to perform adaptation from')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The gpu to run training on')

    # Delete config files
    args = parser.parse_args()
    args.config = args.config[0]
    return args


if __name__ == '__main__':
    main(**vars(parse_args()))
