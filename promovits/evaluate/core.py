"""Objective and subjective evaluation of prosody editing

Files generated during evaluation are saved in eval/. The directory structure
is as follows.

eval
├── objective
│   └── <dataset>
│       └── <condition>
│           └── <partition>
│               └── <speaker>
│                   └── <stem>-<modification>-<ratio>-<feature>.<extension>
└── subjective
    └── <dataset>
        └── <condition>
            └── <partition>
                └── <speaker>
                    └── <stem>-<modification>-<ratio>.wav
"""

import functools
import json
import shutil
import traceback
import warnings

import pyfoal
import pypar
import torch

import promovits


###############################################################################
# Perform evaluation
###############################################################################


# Constant ratios at which we evaluate prosody
RATIOS = [.717, 1.414]


###############################################################################
# Perform evaluation
###############################################################################


def speaker(
    dataset,
    train_partition,
    test_partition,
    checkpoint,
    metrics,
    output_directory,
    log_directory,
    objective_directory,
    subjective_directory,
    gpus=None):
    """Evaluate one adaptation speaker in a dataset"""
    if promovits.MODEL == 'promovits':

        # Perform speaker adaptation and get generator checkpoint
        checkpoint = promovits.train.run(
            dataset,
            checkpoint,
            output_directory,
            log_directory,
            train_partition,
            test_partition,
            True,
            gpus)

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
        input_file = promovits.CACHE_DIR / dataset / f'{stem}-100.wav'
        output_file = \
            original_subjective_directory / f'{stem}-original-100.wav'
        output_file.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(input_file, output_file)

        # Copy prosody and text files
        input_files = [
            path for path in (promovits.CACHE_DIR / dataset).glob(f'{stem}-100*')
            if path.suffix != '.wav']
        for input_file in input_files:
            if input_file.suffix == '.json':
                feature = 'alignment'
            elif input_file.suffix == '.txt':
                feature = 'text'
            else:
                feature = input_file.stem.split('-')[-1]
            output_file = (
                original_objective_directory /
                f'{stem}-original-100-{feature}{input_file.suffix}')
            output_file.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(input_file, output_file)

    # Generate reconstructions
    files = {
        'original': sorted(list(
            original_subjective_directory.rglob('*-original-100.wav')))}
    files['reconstructed'] = sorted([
        subjective_directory / f'{stem}-original-100.wav'
        for stem in test_stems])
    promovits.from_files_to_files(
        files['original'],
        files['reconstructed'],
        checkpoint=checkpoint,
        gpu=None if gpus is None else gpus[0])

    # Copy unchanged prosody features
    for file in files['reconstructed']:
        features = [
            'loudness',
            'periodicity',
            'phonemes',
            'pitch',
            'ppg',
            'text',
            'voicing']
        for feature in features:
            suffix = '.txt' if feature == 'text' else '.pt'
            input_file = (
                original_objective_directory /
                file.parent.name /
                f'{file.stem}-{feature}').with_suffix(suffix)
            output_file = (
                objective_directory /
                file.parent.name /
                f'{file.stem}-{feature}').with_suffix(suffix)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(input_file, output_file)

    # Constant-ratio pitch-shifting
    original_pitch_files = list(
        original_objective_directory.rglob('*-original-100-pitch.pt'))
    for ratio in RATIOS:

        # Shift original pitch and save to disk
        for original_pitch_file in original_pitch_files:
            stem = original_pitch_file.stem[:6]
            key = f'shifted-{int(ratio * 100):03d}'
            pitch = ratio * torch.load(original_pitch_file)
            pitch[pitch < promovits.FMIN] = promovits.FMIN
            pitch[pitch > promovits.FMAX] = promovits.FMAX
            shifted_pitch_file = (
                original_objective_directory /
                original_pitch_file.parent.name /
                f'{stem}-{key}-pitch.pt')
            shifted_pitch_file.parent.mkdir(exist_ok=True, parents=True)
            torch.save(pitch, shifted_pitch_file)

            # Copy unchanged prosody features
            features = [
                'loudness',
                'periodicity',
                'phonemes',
                'ppg',
                'text',
                'voicing']
            for feature in features:
                suffix = '.txt' if feature == 'text' else '.pt'
                input_file = (
                    original_pitch_file.parent /
                    original_pitch_file.name.replace(
                        key, 'original-100').replace(
                        'pitch', feature)).with_suffix(suffix)
                output_file = (
                    shifted_pitch_file.parent /
                    shifted_pitch_file.name.replace(
                        'pitch', feature)).with_suffix(suffix)
                output_file.parent.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(input_file, output_file)

        # Get filenames
        files[key] = [
            file.parent / f'{file.stem[:6]}-{key}.wav'
            for file in files['reconstructed']]
        pitch_files = [
            original_objective_directory/
            file.parent.name /
            f'{file.stem[:6]}-{key}-pitch.pt'
            for file in files['reconstructed']]

        # Generate
        promovits.from_files_to_files(
            files['original'],
            files[key],
            target_pitch_files=pitch_files,
            checkpoint=checkpoint,
            gpu=None if gpus is None else gpus[0])

    # Constant-ratio time-stretching
    original_alignment_files = sorted(
        file for file in
        original_objective_directory.rglob('*-original-100-alignment.json'))
    for ratio in RATIOS:
        key = f'stretched-{int(ratio * 100):03d}'

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
                original_objective_directory /
                original_alignment_file.parent.name /
                f'{original_alignment_file.stem[:6]}-{key}-grid.pt')
            grid_file.parent.mkdir(exist_ok=True, parents=True)
            torch.save(grid, grid_file)

            # Stretch and save other prosody features
            features = [
                'loudness',
                'periodicity',
                'phonemes',
                'pitch',
                'ppg',
                'voicing']
            size = None
            for feature in features:
                input_file = (
                    grid_file.parent /
                    grid_file.name.replace(
                        key, 'original-100').replace('grid', feature))
                output_file = \
                    grid_file.parent / grid_file.name.replace('grid', feature)
                original_feature = torch.load(input_file)
                if size is None:
                    size = original_feature.shape[-1]
                if feature in ['loudness', 'periodicity']:
                    stretched_feature = promovits.interpolate.grid_sample(
                        original_feature.squeeze(),
                        grid.squeeze(),
                        'linear')[None]
                elif feature in ['phonemes', 'voicing']:
                    stretched_feature = promovits.interpolate.grid_sample(
                        original_feature.squeeze(),
                        grid.squeeze(),
                        'nearest')[None]
                elif feature == 'ppg':
                    mode = promovits.PPG_INTERP_METHOD
                    original_feature = torch.nn.functional.interpolate(
                        original_feature[None],
                        size=size,
                        mode=mode,
                        align_corners=None if mode == 'nearest' else False)[0]
                    stretched_feature = promovits.interpolate.ppgs(
                        original_feature.squeeze(),
                        grid.squeeze())
                elif feature == 'pitch':
                    stretched_feature = promovits.interpolate.pitch(
                        original_feature.squeeze(),
                        grid.squeeze())[None]
                torch.save(stretched_feature, output_file)

            # Copy text
            input_file = (
                grid_file.parent /
                grid_file.name.replace(
                    key, 'original-100').replace(
                    'grid', 'text')).with_suffix('.txt')
            output_file = (
                grid_file.parent /
                grid_file.name.replace('grid', 'text')).with_suffix('.txt')
            shutil.copyfile(input_file, output_file)

        # Get filenames
        files[key] = [
            file.parent / f'{file.stem[:6]}-{key}.wav'
            for file in files['reconstructed']]
        grid_files = [
            original_objective_directory /
            file.parent.name /
            f'{file.stem[:6]}-{key}-grid.pt'
            for file in files['reconstructed']]

        # Generate
        promovits.from_files_to_files(
            files['original'],
            files[key],
            grid_files=grid_files,
            checkpoint=checkpoint,
            gpu=None if gpus is None else gpus[0])

    # Constant-ratio loudness-scaling
    original_loudness_files = sorted(list(
        original_objective_directory.rglob('*-original-100-loudness.pt')))
    for ratio in RATIOS:
        key = f'scaled-{int(ratio * 100):03d}'

        # Scale original loudness and save to disk
        for original_loudness_file in original_loudness_files:
            loudness = (
                10 * torch.log2(torch.tensor(ratio)) +
                torch.load(original_loudness_file))
            scaled_loudness_file = (
                original_objective_directory /
                original_loudness_file.parent.name /
                f'{original_loudness_file.stem[:6]}-{key}-loudness.pt')
            torch.save(loudness, scaled_loudness_file)

            # Copy unchanged prosody features
            features = [
                'periodicity',
                'phonemes',
                'pitch',
                'ppg',
                'text',
                'voicing']
            for feature in features:
                suffix = '.txt' if feature == 'text' else '.pt'
                input_file = (
                    original_loudness_file.parent /
                    original_loudness_file.name.replace(
                        key, 'original-100').replace(
                        'loudness', feature)).with_suffix(suffix)
                output_file = (
                    scaled_loudness_file.parent /
                    scaled_loudness_file.name.replace(
                        'loudness', feature)).with_suffix(suffix)
                shutil.copyfile(input_file, output_file)

        # Get filenames
        files[key] = [
            file.parent / f'{file.stem[:6]}-{key}.wav'
            for file in files['reconstructed']]
        loudness_files = [
            original_objective_directory /
            file.parent.name /
            f'{file.stem[:6]}-{key}-loudness.pt'
            for file in files['reconstructed']]

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
            f'{file.stem}-text.txt'
            for file in value]

        # Make sure alignments don't already exist
        for prefix in output_prefixes:
            file = prefix.parent / f'{prefix.stem}-alignment.json'
            file.unlink(missing_ok=True)

        # Ignore warnings when MFA fails, as we retry failures with P2FA
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
        promovits.preprocess.from_files_to_files(
            objective_directory / file.parent.name,
            value,
            text_files,
            gpu=None if gpus is None else gpus[0])

        # Get any failed alignments to retry
        p2fa_args = []
        iterator = zip(text_files, value, output_prefixes)
        for text_file, audio_file, prefix in iterator:
            output_file = prefix.parent / f'{prefix.stem}-alignment.json'
            if not output_file.exists():
                p2fa_args.append((text_file, audio_file, output_file))

        # If any alignments did not succeed with MFA, retry with P2FA
        if p2fa_args:
            with pyfoal.backend('p2fa'):
                pyfoal.from_files_to_files(
                    *zip(*p2fa_args),
                    num_workers=promovits.NUM_WORKERS)

                # Convert alignments to phoneme indices
                for file in [arg[2] for arg in p2fa_args]:

                    # Load alignment
                    alignment = pypar.Alignment(file)

                    # Get times to sample phonemes
                    pitch = torch.load(
                        file.parent / file.name.replace(
                            'alignment.json', 'pitch.pt'))
                    hopsize = promovits.HOPSIZE / promovits.SAMPLE_RATE
                    times = torch.arange(pitch.shape[1]) * hopsize

                    # Convert to indices
                    indices = pyfoal.convert.alignment_to_indices(
                        alignment,
                        hopsize,
                        times=times)

                    # Save indices
                    indices_file = (
                        file.parent /
                        f'{file.stem.replace("alignment", "phonemes")}.pt')
                    torch.save(
                        torch.tensor(indices, dtype=torch.long)[None],
                        indices_file)

    # Perform objective evaluation
    speaker_metrics = default_metrics(gpus)
    results = {'objective': {'raw': {}}}
    for key, value in files.items():
        results['objective']['raw'][key] = []

        for file in value:

            # Get prosody metrics
            gpu = None if gpus is None else gpus[0]
            file_metrics = promovits.evaluate.metrics.Metrics(gpu)

            # Get target filepath
            target_prefix = \
                original_objective_directory / file.parent.name / file.stem

            # Get predicted filepath
            predicted_prefix = \
                objective_directory / file.parent.name / file.stem

            # Update metrics
            try:
                prosody_args = (
                    torch.load(f'{predicted_prefix}-pitch.pt'),
                    torch.load(f'{predicted_prefix}-periodicity.pt'),
                    torch.load(f'{predicted_prefix}-loudness.pt'),
                    torch.load(f'{predicted_prefix}-voicing.pt'),
                    torch.load(f'{target_prefix}-pitch.pt'),
                    torch.load(f'{target_prefix}-periodicity.pt'),
                    torch.load(f'{target_prefix}-loudness.pt'),
                    torch.load(f'{target_prefix}-voicing.pt'),
                    torch.load(f'{predicted_prefix}-phonemes.pt'),
                    torch.load(f'{target_prefix}-phonemes.pt'))

                # TODO - Stretched target ppgs are already resampled. Others are not

                # Get predicted PPGs
                size = prosody_args[0].shape[-1]
                mode = promovits.PPG_INTERP_METHOD
                predicted_ppgs = torch.nn.functional.interpolate(
                    torch.load(f'{predicted_prefix}-ppg.pt')[None],
                    size=size,
                    mode=mode,
                    align_corners=None if mode == 'nearest' else False)[0]

                # Get target PPGs
                target_ppgs = torch.nn.functional.interpolate(
                    torch.load(f'{target_prefix}-ppg.pt')[None],
                    size=size,
                    mode=mode,
                    align_corners=None if mode == 'nearest' else False)[0]

                ppg_args = (predicted_ppgs, target_ppgs)

                condition = '-'.join(target_prefix.stem.split('-')[1:3])
                metrics[condition].update(prosody_args, ppg_args)
                speaker_metrics[condition].update(prosody_args, ppg_args)
                file_metrics.update(prosody_args, ppg_args)
            except Exception as error:
                print(traceback.format_exc())
                import pdb; pdb.set_trace()
                pass

            # Get results for this file
            results['objective']['raw'][key].append(
                (file.stem, file_metrics()))

            # Reset prosody metrics
            file_metrics.reset()

    # Get the total number of samples we have generated
    files = subjective_directory.rglob('*.wav')
    results['num_samples'] = sum([file.stat().st_size for file in files]) // 4
    results['num_frames'] = results['num_samples'] // promovits.HOPSIZE

    # Get results for this speaker
    results['objective']['average'] = {
        key: value() for key, value in speaker_metrics.items()}

    # Print results and save to disk
    print(results)
    with open(objective_directory / 'results.json', 'w') as file:
        json.dump(results, file, indent=4, sort_keys=True)


def datasets(datasets, checkpoint=None, gpus=None):
    """Evaluate the performance of the model on datasets"""
    # Turn on benchmarking
    current_benchmark = promovits.BENCHMARK
    promovits.BENCHMARK = True

    # Evaluate on each dataset
    for dataset in datasets:

        # Reset benchmarking
        promovits.TIMER.reset()

        # Get adaptation partitions for this dataset
        partitions = promovits.load.partition(dataset)
        train_partitions = sorted(list(
            partition for partition in partitions.keys()
            if 'train-adapt' in partition))
        test_partitions = sorted(list(
            partition for partition in partitions.keys()
            if 'test-adapt' in partition))

        # Prosody metrics
        metrics = default_metrics(gpus)

        # Evaluate on each partition
        iterator = zip(train_partitions, test_partitions)
        for train_partition, test_partition in iterator:

            # Index of this adaptation partition
            index = train_partition.split('-')[-1]

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
                metrics,
                adapt_directory,
                adapt_directory,
                objective_directory,
                subjective_directory,
                gpus)

        # Aggregate results
        results_directory = (
            promovits.EVAL_DIR /
            'objective' /
            dataset /
            promovits.CONFIG)
        results = {
            'num_samples': 0,
            'num_frames': 0,
            'prosody': {key: value() for key, value in metrics.items()}}
        for file in results_directory.rglob('results.json'):
            with open(file) as file:
                result = json.load(file)
            results['num_samples'] += result['num_samples']
            results['num_frames'] += result['num_frames']

        # Parse benchmarking results
        results['benchmark'] = {'raw': promovits.TIMER()}

        # Average benchmarking over samples
        results['benchmark']['average'] = {
            key: value / results['num_samples']
            for key, value in results['benchmark']['raw'].items()}

        # Print results and save to disk
        print(results)
        with open(results_directory / 'results.json', 'w') as file:
            json.dump(results, file, indent=4, sort_keys=True)

    # Maybe turn off benchmarking
    promovits.BENCHMARK = current_benchmark


###############################################################################
# Utilities
###############################################################################


def default_metrics(gpus):
    """Construct the default metrics dictionary for each condition"""
    # Bind shared parameter
    gpu = None if gpus is None else gpus[0]
    metric_fn = functools.partial(promovits.evaluate.metrics.Metrics, gpu)

    # Create metric object for each condition
    metrics = {'original-100': metric_fn()}
    for condition in ['scaled', 'shifted', 'stretched']:
        for ratio in RATIOS:
            metrics[f'{condition}-{int(ratio * 100):03d}'] = metric_fn()

    return metrics


def load_ppgs(file, size):
    """Load and interpolate PPGs"""
    mode = promovits.PPG_INTERP_METHOD
    return torch.nn.functional.interpolate(
        torch.load(file)[None],
        size=size,
        mode=mode,
        align_corners=None if mode == 'nearest' else False)[0]
