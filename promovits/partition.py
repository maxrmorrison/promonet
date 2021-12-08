import argparse
import json
import random

import promovits


###############################################################################
# Partition
###############################################################################


def dataset(name):
    """Partition datasets and save partitions to disk"""
    # Handle vctk
    if name == 'vctk':
        return vctk()

    # Handle daps
    if name == 'daps':
        return daps()


def daps():
    """Partition the DAPS dataset"""
    # Get stems
    directory = promovits.CACHE_DIR / 'daps'
    stems = [file.stem for file in directory.glob('*.wav')]

    # We manually select test speakers to ensure gender balance
    test_speakers = [
        # Female
        '0002',
        '0007',
        '0010',
        '0013',
        '0019',

        # Male
        '0003',
        '0005',
        '0014',
        '0015',
        '0017']

    # Get test partition indices
    test_stems = [
        stem for stem in stems if stem.split('-')[0] in test_speakers]

    # Shuffle so adjacent indices aren't always same speaker
    random.seed(promovits.RANDOM_SEED)
    random.shuffle(test_stems)

    # Get residual indices
    valid_stems = [stem for stem in stems if stem not in test_stems]
    random.shuffle(valid_stems)

    # Daps is eval-only
    return {'train': [], 'valid': valid_stems, 'test': test_stems}


def vctk():
    """Partition the vctk dataset"""
    # Get list of speakers
    directory = promovits.CACHE_DIR / 'vctk'
    stems = [file.stem for file in directory.glob('*.wav')]

    # We manually select test speakers to ensure gender balance
    test_speakers = [
        # Female
        '0013',
        '0037',
        '0070',
        '0082',
        '0108',

        # Male
        '0016',
        '0032',
        '0047',
        '0073',
        '0083']

    # Get test partition indices
    test_stems = [
        stem for stem in stems if stem.split('-')[0] in test_speakers]

    # Shuffle so adjacent indices aren't always same speaker
    random.seed(promovits.RANDOM_SEED)
    random.shuffle(test_stems)

    # Get residual indices
    residual = [stem for stem in stems if stem not in test_stems]
    random.shuffle(residual)

    # Split into train/valid
    split = int(.95 * len(residual))
    train_stems = residual[:split]
    valid_stems = residual[split:]

    return {'train': train_stems, 'valid': valid_stems, 'test': test_stems}


###############################################################################
# Entry point
###############################################################################


def main(datasets, overwrite):
    """Partition datasets and save to disk"""
    for name in datasets:

        # Check if partition already exists
        file = promovits.PARTITION_DIR / f'{name}.json'
        if file.exists():
            if not overwrite:
                print(f'Not overwriting existing partition {file}')
                continue

        # Save to disk
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, 'w') as file:
            json.dump(dataset(name), file, ensure_ascii=False, indent=4)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='The datasets to partition')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Whether to overwrite existing partitions')
    return parser.parse_args()


if __name__ == '__main__':
    main(**vars(parse_args()))
