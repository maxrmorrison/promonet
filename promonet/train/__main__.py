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
    adapt=False,
    gpus=None):
    # Create output directory
    directory = promonet.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    promonet.train.run(
        dataset,
        directory,
        directory,
        directory,
        train_partition,
        valid_partition,
        adapt,
        gpus)

    # Evaluate
    # TEMPORARY - only evaluate on training dataset for now
    # promonet.evaluate.datasets(promonet.DATASETS, directory, gpus)
    promonet.evaluate.datasets([dataset], directory, gpus)


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
        default='vctk',
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
        '--adapt',
        action='store_true',
        help='Whether to use hyperparameters for speaker adaptation')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='The gpus to run training on')

    # Delete config files
    args = parser.parse_args()
    args.config = args.config[0]
    return args


if __name__ == '__main__':
    main(**vars(parse_args()))
