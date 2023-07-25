"""
Data partitions

DAPS
===
* train_adapt_{:02d} - Training dataset for speaker adaptation (10 speakers)
* test_adapt_{:02d} - Test dataset for speaker adaptation
    (10 speakers; 10 examples per speaker; 4-10 seconds)

LibriTTS
====
* train - Training data
* valid - Validation set of seen speakers for debugging and tensorboard
    (10 examples)
* train_adapt_{:02d} - Training dataset for speaker adaptation (10 speakers)
* test_adapt_{:02d} - Test dataset for speaker adaptation
    (10 speakers; 10 examples per speaker; 4-10 seconds)

VCTK
====
* train - Training data
* valid - Validation set of seen speakers for debugging and tensorboard
    (10 examples; 4-10 seconds)
* train_adapt_{:02d} - Training dataset for speaker adaptation (10 speakers)
* test_adapt_{:02d} - Test dataset for speaker adaptation
    (10 speakers; 10 examples per speaker; 4-10 seconds)
"""
import os
import functools
import itertools
import json
import random

import torchaudio

import promonet


###############################################################################
# Constants
###############################################################################


# Range of allowable test sample lengths in seconds
MAX_TEST_SAMPLE_LENGTH = 10.
MIN_TEST_SAMPLE_LENGTH = 4.


###############################################################################
# Adaptation speaker IDs
###############################################################################


# We manually select test speakers to ensure gender balance
DAPS_ADAPTATION_SPEAKERS = [
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

# Speakers selected by sorting the train-clean-100 speakers by longest total
# recording duration and manually selecting speakers with more natural,
# conversational (as opposed to read) prosody
LIBRITTS_ADAPTATION_SPEAKERS = [
    # Female
    '0040',
    '0669',
    '4362',
    '5022',
    '8123',

    # Male
    '0196',
    '0460',
    '1355',
    '3664',
    '7067']

# Gender-balanced VCTK speakers
VCTK_ADAPTATION_SPEAKERS = [
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


###############################################################################
# Partition
###############################################################################


def adaptation(name):
    """Partition dataset for speaker adaptation"""
    directory = promonet.CACHE_DIR / name
    train = [
        f'{file.parent.name}/{file.stem}'
        for file in directory.rglob('*.wav')]
    return {'train': train, 'valid': []}


def datasets(datasets):
    """Partition datasets and save to disk"""
    for name in datasets:

        metadata_files = (promonet.CACHE_DIR / name).glob('-metadata.json')
        for metadata_file in metadata_files:
            os.remove(metadata_file)

        # Partition
        if name == 'vctk':
            partition = vctk()
        elif name == 'daps':
            partition = daps()

        # All other datasets are assumed to be for speaker adaptation
        else:
            partition = adaptation(name)

        # Sort partitions
        partition = {key: sorted(value) for key, value in partition.items()}

        # Save to disk
        file = promonet.PARTITION_DIR / f'{name}.json'
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, 'w') as file:
            json.dump(partition, file, indent=4)


def daps():
    """Partition the DAPS dataset"""
    # Get stems
    directory = promonet.CACHE_DIR / 'daps'
    stems = [
        f'{file.parent.name}/{file.stem}'
        for file in directory.rglob('*.txt')]

    # Create speaker adaptation partitions
    return adaptation_partitions(
        directory,
        stems,
        DAPS_ADAPTATION_SPEAKERS)


def libritts():
    """Partition libritts dataset"""
    # Get list of speakers
    directory = promonet.CACHE_DIR / 'libritts'
    stems = {
        f'{file.parent.name}/{file.stem}'
        for file in directory.rglob('*.TextGrid')}

    # Get speaker map
    with open(directory / 'speakers.json') as file:
        speaker_map = json.load(file)

    # Get adaptation speakers
    speakers = [
        f'{speaker_map[speaker][0]:04d}'
        for speaker in LIBRITTS_ADAPTATION_SPEAKERS]

    # Create speaker adaptation partitions
    adapt_partitions = adaptation_partitions(
        directory,
        stems,
        speakers)

    # Get test partition indices
    test_stems = list(
        itertools.chain.from_iterable(adapt_partitions.values()))

    # Get residual indices
    residual = [stem for stem in stems if stem not in test_stems]
    random.shuffle(residual)

    # Get validation stems
    filter_fn = functools.partial(meets_length_criteria, directory)
    valid_stems = list(filter(filter_fn, residual))[:10]

    # Get training stems
    train_stems = [stem for stem in residual if stem not in valid_stems]

    # Merge training and adaptation partitions
    partition = {'train': sorted(train_stems), 'valid': sorted(valid_stems)}
    return {**partition, **adapt_partitions}


def vctk():
    """Partition the vctk dataset"""
    # Get list of speakers
    directory = promonet.CACHE_DIR / 'vctk'
    stems = {
        f'{file.parent.name}/{file.stem}'
        for file in directory.rglob('*.txt')}

    # Create speaker adaptation partitions
    adapt_partitions = adaptation_partitions(
        directory,
        stems,
        VCTK_ADAPTATION_SPEAKERS)

    # Get test partition indices
    test_stems = list(
        itertools.chain.from_iterable(adapt_partitions.values()))

    # Get residual indices
    residual = [stem for stem in stems if stem not in test_stems]
    random.shuffle(residual)

    # Get validation stems
    filter_fn = functools.partial(meets_length_criteria, directory)
    valid_stems = list(filter(filter_fn, residual))[:10]

    # Get training stems
    train_stems = [stem for stem in residual if stem not in valid_stems]

    # Merge training and adaptation partitions
    partition = {'train': train_stems, 'valid': valid_stems}
    return {**partition, **adapt_partitions}


###############################################################################
# Utilities
###############################################################################


def adaptation_partitions(directory, stems, speakers):
    """Create the speaker adaptation partitions"""
    # Get adaptation data
    adaptation_stems = {
        speaker: [stem for stem in stems if stem.split('/')[0] == speaker]
        for speaker in speakers}

    # Get length filter
    filter_fn = functools.partial(meets_length_criteria, directory)

    # Partition adaptation data
    adaptation_partition = {}
    random.seed(promonet.RANDOM_SEED)
    for i, speaker in enumerate(speakers):
        random.shuffle(adaptation_stems[speaker])

        # Partition speaker data
        test_adapt_stems = list(
            filter(filter_fn, adaptation_stems[speaker]))[:10]
        train_adapt_stems = [
            stem for stem in adaptation_stems[speaker]
            if stem not in test_adapt_stems]

        # Save partition
        adaptation_partition[f'train-adapt-{i:02d}'] = train_adapt_stems
        adaptation_partition[f'test-adapt-{i:02d}'] = test_adapt_stems

    return adaptation_partition


def meets_length_criteria(directory, stem):
    """Returns True if the audio file duration is within the length criteria"""
    info = torchaudio.info(directory / f'{stem}.wav')
    duration = info.num_frames / info.sample_rate
    return MIN_TEST_SAMPLE_LENGTH <= duration <= MAX_TEST_SAMPLE_LENGTH
