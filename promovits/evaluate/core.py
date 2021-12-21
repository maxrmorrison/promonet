import json
import shutil

import pyfoal
import pysodic
import torch

import promovits


###############################################################################
# Perform evaluation
###############################################################################


# Constant ratios at which we evaluate prosody
RATIOS = [.5, .717, 1.414, 2.]


###############################################################################
# Perform evaluation
###############################################################################


def speaker(
    dataset,
    train_partition,
    test_partition,
    adapt_directory,
    objective_directory,
    subjective_directory,
    gpus=None):
    """Evaluate one adaptation speaker in a dataset"""
    # TODO - add cache checks throughout

    # Turn on benchmarking
    current_benchmark = promovits.BENCHMARK
    promovits.BENCHMARK = True

    # Perform speaker adaptation
    promovits.train.run(
        dataset,
        adapt_directory,
        train_partition,
        test_partition,
        adapt_directory.parent.parent.parent,
        True,
        gpus)

    # Get latest checkpoint
    adapt_checkpoint = promovits.latest_checkpoint_path(adapt_directory)

    # Directory to save original audio files
    original_subjective_directory = (
        subjective_directory.parent.parent /
        'original' /
        subjective_directory.stem)
    original_subjective_directory.mkdir(exist_ok=True, parents=True)

    # Directory to save original prosody files
    original_objective_directory = (
        objective_directory.parent.parent /
        'original' /
        objective_directory.stem)
    original_objective_directory.mkdir(exist_ok=True, parents=True)

    # Stems to use for evaluation
    test_stems = sorted(promovits.load.partition(dataset)[test_partition])

    # Copy original files
    for stem in test_stems:

        # Copy audio files
        input_file = promovits.CACHE_DIR / dataset / f'{stem}.wav'
        output_file = original_subjective_directory / input_file.name
        shutil.copyfile(input_file, output_file)

        # Copy text files
        input_file = promovits.CACHE_DIR / dataset / f'{stem}.txt'
        output_file = original_objective_directory / input_file.name
        shutil.copyfile(input_file, output_file)

        # Copy prosody files
        input_files = [
            path for path in (promovits.CACHE_DIR / dataset).glob(f'{stem}*')
            if path.suffix != '.wav']
        for input_file in input_files:
            output_file = original_objective_directory / input_file.name
            shutil.copyfile(input_file, output_file)

    # Generate reconstructions
    files = {
        'original': list(original_subjective_directory.glob('*.wav')),
        'reconstructed':
            [subjective_directory / file.name for file in input_files]}
    promovits.from_files_to_files(
        files['original'],
        files['reconstructed'],
        checkpoint=adapt_checkpoint,
        gpu=None if gpus is None else gpus[0])

    # Constant-ratio pitch-shifting
    original_pitch_files = original_objective_directory.glob('*-pitch.pt')
    for ratio in RATIOS:
        key = f'pitch-{int(ratio * 100):03d}'

        # Shift original pitch and save to disk
        for original_pitch_file in original_pitch_files:
            pitch = ratio * torch.load(original_pitch_file)
            shifted_pitch_file = (
                objective_directory /
                f'{original_pitch_file.stem[:-6]}-{key}.pt')
            torch.save(pitch, shifted_pitch_file)

        # Get filenames
        files[key] = [
            f'{file.stem}-{key}.wav' for file in files['reconstructed']]
        pitch_files = [file.with_suffix('.pt') for file in files[key]]

        # Generate
        promovits.from_files_to_files(
            input_files,
            files[key],
            target_pitch_files=pitch_files,
            checkpoint=adapt_checkpoint,
            gpu=None if gpus is None else gpus[0])

    # Constant-ratio time-stretching
    original_phoneme_files = original_objective_directory.glob('*-phonemes.pt')
    for ratio in RATIOS:
        key = f'duration-{int(ratio * 100):03d}'

        # Stretch original alignment and save to disk
        for original_phoneme_file in original_phoneme_files:
            phonemes = torch.load(original_phoneme_file)

            # TODO - interpolation
            phonemes = pyfoal.interpolate_vowels(phonemes.numpy(), ratio)

            # TODO - convert to alignment and save alignment
            alignment = pyfoal.convert.indices_to_alignment(phonemes)
            stretched_alignment_file = (
                objective_directory /
                f'{original_phoneme_file.stem[:-9]}-{key}.json')
            alignment.save(stretched_alignment_file)

        # Get filenames
        files[key] = [
            f'{file.stem}-{key}.wav' for file in files['reconstructed']]
        alignment_files = [file.with_suffix('.json') for file in files[key]]

        # Generate
        promovits.from_files_to_files(
            input_files,
            files[key],
            target_alignment_files=alignment_files,
            checkpoint=adapt_checkpoint,
            gpu=None if gpus is None else gpus[0])

    # Constant-ratio loudness-scaling
    original_loudness_files = \
        original_objective_directory.glob('*-loudness.pt')
    for ratio in RATIOS:
        key = f'loudness-{int(ratio * 100):03d}'

        # Scale original loudness and save to disk
        for original_loudness_file in original_loudness_files:
            loudness = \
                10 * torch.log2(ratio) + torch.load(original_loudness_file)
            scaled_loudness_file = (
                objective_directory /
                f'{original_loudness_file.stem[:-9]}-{key}.pt')
            torch.save(loudness, scaled_loudness_file)

        # Get filenames
        files[key] = [
            f'{file.stem}-{key}.wav' for file in files['reconstructed']]
        loudness_files = [file.with_suffix('.pt') for file in files[key]]

        # Generate
        promovits.from_files_to_files(
            input_files,
            files[key],
            target_loudness_files=loudness_files,
            checkpoint=adapt_checkpoint,
            gpu=None if gpus is None else gpus[0])

    # Extract prosody from generated files
    for key, value in files:
        output_prefixes = [objective_directory / file.stem for file in value]
        text_files = [
            original_objective_directory / ''.join(file.stem.split('-')[:2])
            for file in value]
        pysodic.from_files_to_files(
            value,
            output_prefixes,
            text_files,
            promovits.HOPSIZE / promovits.SAMPLE_RATE,
            promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
            None if gpus is None else gpus[0])

    # Perform objective evaluation
    results = {}
    for key, value in files:
        predicted_prefixes = []
        target_prefixes = []
        results[key] = pysodic.evaluate.from_files(
            predicted_prefixes,
            target_prefixes,
            None if gpus is None else gpus[0])

    # Save to disk
    output_file = objective_directory / 'results.json'
    with open(output_file, 'w') as file:
        json.dump(results, output_file, indent=4, sort_keys=True)

    # TODO - Save benchmarking to disk
    # TODO - get number of samples via glob of wav file sizes
    print(promovits.TIMER)
    promovits.TIMER.save(objective_directory / 'benchmark.json')

    # Maybe turn off benchmarking
    promovits.BENCHMARK = current_benchmark


def datasets(config, datasets, gpus=None):
    """Evaluate the performance of the model on datasets"""
    # Evaluate on each dataset
    for dataset in datasets:

        # Get adaptation partitions for this dataset
        partitions = promovits.load.partition(dataset)
        train_partitions = sorted(list(
            partition for partition in partitions.keys()
            if 'train_adapt' in partition))
        test_partitions = sorted(list(
            partition for partition in partition.keys()
            if 'test_adapt' in partition))

        # Evaluate on each partition
        iterator = zip(train_partitions, test_partitions)
        for train_partition, test_partition in iterator:

            # Index of this adaptation partition
            index = train_partition.split('-')[-1]

            # Output directory for adaptation artifacts
            adapt_directory = (
                promovits.RUNS_DIR /
                config.stem /
                'adapt' /
                dataset /
                index)

            # Output directory for objective evaluation
            objective_directory = (
                promovits.EVAL_DIR /
                'objective' /
                dataset /
                config.stem /
                index)

            # Output directory for subjective evaluation
            subjective_directory = (
                promovits.EVAL_DIR /
                'subjective' /
                dataset /
                config.stem /
                index)

            # Evaluate a speaker
            speaker(
                dataset,
                train_partition,
                test_partition,
                adapt_directory,
                objective_directory,
                subjective_directory,
                gpus)
