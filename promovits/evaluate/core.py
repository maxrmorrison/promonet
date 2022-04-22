import json
import shutil

import pyfoal
import pypar
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
    checkpoint,
    output_directory,
    log_directory,
    objective_directory,
    subjective_directory,
    gpus=None):
    """Evaluate one adaptation speaker in a dataset"""
    # Turn on benchmarking
    current_benchmark = promovits.BENCHMARK
    promovits.BENCHMARK = True

    if promovits.MODEL == 'promovits':

        # Perform speaker adaptation
        promovits.train.run(
            dataset,
            checkpoint,
            output_directory,
            log_directory,
            train_partition,
            test_partition,
            True,
            gpus)

        # Get latest generator checkpoint
        checkpoint = promovits.latest_checkpoint_path(output_directory)

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
        output_file = original_subjective_directory / f'{stem}.wav'
        output_file.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(input_file, output_file)

        # Copy prosody and text files
        input_files = [
            path for path in (promovits.CACHE_DIR / dataset).glob(f'{stem}*')
            if path.suffix != '.wav']
        for input_file in input_files:
            output_file = (
                original_objective_directory /
                input_file.parent.name /
                input_file.name)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(input_file, output_file)

    # Generate reconstructions
    files = {
        'original': sorted(list(
            original_subjective_directory.rglob('*.wav')))}
    files['reconstructed'] = sorted(list(
        [subjective_directory / f'{stem}.wav' for stem in test_stems]))
    promovits.from_files_to_files(
        files['original'],
        files['reconstructed'],
        checkpoint=checkpoint,
        gpu=None if gpus is None else gpus[0])

    # Constant-ratio pitch-shifting
    original_pitch_files = list(original_objective_directory.rglob('*-pitch.pt'))
    for ratio in RATIOS:
        key = f'pitch-{int(ratio * 100):03d}'

        # Shift original pitch and save to disk
        for original_pitch_file in original_pitch_files:
            pitch = ratio * torch.load(original_pitch_file)
            shifted_pitch_file = (
                objective_directory /
                original_pitch_file.parent.name /
                f'{original_pitch_file.stem[:-6]}-{key}.pt')
            shifted_pitch_file.parent.mkdir(exist_ok=True, parents=True)
            torch.save(pitch, shifted_pitch_file)

        # Get filenames
        files[key] = [
            file.parent / f'{file.stem}-{key}.wav'
            for file in files['reconstructed']]
        pitch_files = [
            objective_directory/
            file.parent.name /
            f'{file.stem}-{key}.pt' for file in files['reconstructed']]

        # Generate
        promovits.from_files_to_files(
            files['original'],
            files[key],
            target_pitch_files=pitch_files,
            checkpoint=checkpoint,
            gpu=None if gpus is None else gpus[0])

    # Constant-ratio time-stretching
    original_alignment_files = sorted(list(
        original_objective_directory.rglob('*.json')))
    for ratio in RATIOS:
        key = f'duration-{int(ratio * 100):03d}'

        # Stretch original alignment and save to disk
        for original_alignment_file in original_alignment_files:

            # Load alignment
            alignment = pypar.Alignment(original_alignment_file)

            # Interpolate voiced regions
            interpolated = pyfoal.interpolate.voiced(alignment, ratio)
            grid = promovits.interpolate.grid.from_alignments(
                alignment,
                interpolated)

            # Save alignment to disk
            grid_file = (
                objective_directory /
                original_alignment_file.parent.name /
                f'{original_alignment_file.stem}-{key}.pt')
            grid_file.parent.mkdir(exist_ok=True, parents=True)
            torch.save(grid, grid_file)

        # Get filenames
        files[key] = [
            file.parent / f'{file.stem}-{key}.wav'
            for file in files['reconstructed']]
        grid_files = [
            objective_directory /
            file.parent.name /
            f'{file.stem}-{key}.pt' for file in files['reconstructed']]

        # Generate
        promovits.from_files_to_files(
            files['original'],
            files[key],
            grid_files=grid_files,
            checkpoint=checkpoint,
            gpu=None if gpus is None else gpus[0])

    # Constant-ratio loudness-scaling
    original_loudness_files = sorted(list(
        original_objective_directory.rglob('*-loudness.pt')))
    for ratio in RATIOS:
        key = f'loudness-{int(ratio * 100):03d}'

        # Scale original loudness and save to disk
        for original_loudness_file in original_loudness_files:
            loudness = (
                10 * torch.log2(torch.tensor(ratio)) +
                torch.load(original_loudness_file))
            scaled_loudness_file = (
                objective_directory /
                original_loudness_file.parent.name /
                f'{original_loudness_file.stem[:-9]}-{key}.pt')
            torch.save(loudness, scaled_loudness_file)

        # Get filenames
        files[key] = [
            file.parent / f'{file.stem}-{key}.wav'
            for file in files['reconstructed']]
        loudness_files = [
            objective_directory /
            file.parent.name /
            f'{file.stem}-{key}.pt' for file in files['reconstructed']]

        # Generate
        promovits.from_files_to_files(
            files['original'],
            files[key],
            target_loudness_files=loudness_files,
            checkpoint=checkpoint,
            gpu=None if gpus is None else gpus[0])

    # Extract prosody from generated files
    for key, value in files.items():
        output_prefixes = [
            objective_directory / file.parent.name / file.stem
            for file in value]
        text_files = [
            original_objective_directory /
                file.parent.name /
                f'{file.stem}.txt'
            for file in value]
        pysodic.from_files_to_files(
            value,
            output_prefixes,
            text_files,
            promovits.HOPSIZE / promovits.SAMPLE_RATE,
            promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
            None if gpus is None else gpus[0])

    # Perform objective evaluation
    results = {'objective': {'raw': {}}}
    for key, value in files.items():
        predicted_prefixes = []
        target_prefixes = []
        results['objective']['raw'][key] = pysodic.evaluate.from_files(
            predicted_prefixes,
            target_prefixes,
            None if gpus is None else gpus[0])

    # Get the total number of samples we have generated
    files = subjective_directory.rglob('*.wav')
    results['num_samples'] = sum([file.stat().st_size for file in files]) // 4
    results['num_frames'] = results['num_samples'] // promovits.HOPSIZE

    # Average objective evaluation over frames
    results['objective']['average'] = {
        result / results['num_frames']
        for key, result in results['objective']['average']}

    # Parse benchmarking results
    results['benchmark'] = {'raw': promovits.TIMER()}

    # Average benchmarking over samples
    results['benchmark']['average'] = [
        result / results['num_samples']
        for key, result in results['benchmark']['raw']
        if key != 'load']

    # Print results and save to disk
    print(results)
    with open(objective_directory / 'results.json', 'w') as file:
        json.dump(results, file, indent=4, sort_keys=True)

    # Maybe turn off benchmarking
    promovits.BENCHMARK = current_benchmark


def datasets(datasets, checkpoint=None, gpus=None):
    """Evaluate the performance of the model on datasets"""
    # Evaluate on each dataset
    for dataset in datasets:

        # Get adaptation partitions for this dataset
        partitions = promovits.load.partition(dataset)
        train_partitions = sorted(list(
            partition for partition in partitions.keys()
            if 'train-adapt' in partition))
        test_partitions = sorted(list(
            partition for partition in partitions.keys()
            if 'test-adapt' in partition))

        # Evaluate on each partition
        iterator = zip(train_partitions, test_partitions)
        for train_partition, test_partition in iterator:

            # Index of this adaptation partition
            index = train_partition.split('-')[-1]

            # Directory containing checkpoint to load from
            if promovits.MODEL == 'promovits':
                checkpoint = promovits.RUNS_DIR / promovits.CONFIG

            # Output directory for checkpoints and logs
            adapt_directory = (
                promovits.RUNS_DIR /
                promovits.CONFIG /
                'adapt' /
                dataset /
                index)

            # Output directory for objective evaluation
            objective_directory = (
                promovits.EVAL_DIR /
                'objective' /
                dataset /
                promovits.CONFIG /
                index)

            # Output directory for subjective evaluation
            subjective_directory = (
                promovits.EVAL_DIR /
                'subjective' /
                dataset /
                promovits.CONFIG /
                index)

            # Evaluate a speaker
            speaker(
                dataset,
                train_partition,
                test_partition,
                checkpoint,
                adapt_directory,
                adapt_directory,
                objective_directory,
                subjective_directory,
                gpus)

        # Aggregate objective and benchmarking results
        results_directory = (
            promovits.EVAL_DIR /
            'objective' /
            dataset /
            promovits.CONFIG)
        results_files = results_directory.rglob('results.json')
        results = {'objective': {'raw': {}}, 'benchmark': {}}
        for file in results_files:
            with open(file) as file:
                result = json.load(file)
            for key, value in result['objective']['raw']:
                results['objective']['raw'][key] += value
            for key, value in result['benchmark']['raw']:
                results['benchmark']['raw'][key] += value
            results['num_samples'] += result['num_samples']
            results['num_frames'] += result['num_frames']

        # Print results and save to disk
        print(results)
        with open(results_directory / 'results.json', 'w') as file:
            json.dump(results, file, indent=4, sort_keys=True)
