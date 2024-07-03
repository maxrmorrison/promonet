"""Objective and subjective evaluation of prosody editing

Files generated during evaluation are saved in eval/. The directory structure
is as follows.

eval
├── objective
|   └── <condition>
|       └── <dataset>-<speaker>-<utterance>-<modification>-<ratio>-<feature>.<extension>
└── subjective
    └── <condition>
        └── <dataset>-<speaker>-<utterance>-<modification>-<ratio>.wav

Results are saved in results/. The directory structure is as follows.

results
└── <condition>
    ├── <dataset>
    |   ├── <speaker>.json  # Per-speaker and per-file results
    |   ├── results.json    # Overall results for this dataset
    |   └── speaker.pdf     # Speaker cluster plot
    └── results.json        # Overall results across datasets
"""
import functools
import json
import shutil

import ppgs
import torch
import torchaudio
import torchutil

import promonet


###############################################################################
# Perform evaluation
###############################################################################


@torchutil.notify('evaluate')
def datasets(datasets, adapt=promonet.ADAPTATION, gpu=None):
    """Evaluate the performance of the model on datasets"""
    aggregate_metrics = default_metrics()

    # Evaluate on each dataset
    for dataset in datasets:

        # Reset benchmarking
        torchutil.time.reset()

        # Get adaptation partitions for this dataset
        partitions = promonet.load.partition(dataset, adapt)
        if adapt:
            train_partitions = sorted(list(
                partition for partition in partitions.keys()
                if 'train-adapt' in partition))
            test_partitions = sorted(list(
                partition for partition in partitions.keys()
                if 'test-adapt' in partition))
        else:
            test_partitions = sorted(list(
                partition for partition in partitions.keys()
                if 'test' in partition))
            train_partitions = [None] * len(test_partitions)

        # Per-dataset metrics
        dataset_metrics = default_metrics()

        # Evaluate on each partition
        iterator = zip(train_partitions, test_partitions)
        for train_partition, test_partition in iterator:

            # Iterate over speakers
            indices = list(set(
                [stem.split('/')[0] for stem in partitions[test_partition]]))
            for index in indices:

                # Output directory for checkpoints and logs
                checkpoint_directory = promonet.RUNS_DIR / promonet.CONFIG

                # Output directory for objective evaluation
                objective_directory = (
                    promonet.EVAL_DIR /
                    'objective' /
                    promonet.CONFIG)
                objective_directory.mkdir(exist_ok=True, parents=True)

                # Output directory for subjective evaluation
                subjective_directory = (
                    promonet.EVAL_DIR /
                    'subjective' /
                    promonet.CONFIG)
                subjective_directory.mkdir(exist_ok=True, parents=True)

                # Evaluate a speaker
                speaker(
                    dataset,
                    train_partition,
                    test_partition,
                    aggregate_metrics,
                    dataset_metrics,
                    checkpoint_directory,
                    objective_directory,
                    subjective_directory,
                    index,
                    adapt,
                    gpu)

        # Aggregate results
        results_directory = promonet.RESULTS_DIR / promonet.CONFIG / dataset
        results_directory.mkdir(exist_ok=True, parents=True)
        results = {'num_samples': 0, 'num_frames': 0}
        results |= {key: value() for key, value in dataset_metrics.items()}

        results_directory = promonet.RESULTS_DIR / promonet.CONFIG / dataset
        for file in results_directory.glob(f'*.json'):
            if file.stem == 'results':
                continue
            with open(file) as file:
                result = json.load(file)
            results['num_samples'] += result['num_samples']
            results['num_frames'] += result['num_frames']

        # Parse benchmarking results
        results['benchmark'] = {'raw': torchutil.time.results()}
        results['benchmark']['rtf'] = {
            key: (results['num_samples'] / promonet.SAMPLE_RATE) / value
            for key, value in results['benchmark']['raw'].items()}

        # Print results and save to disk
        print(json.dumps(results, indent=4, sort_keys=True))
        with open(results_directory / f'results.json', 'w') as file:
            json.dump(results, file, indent=4, sort_keys=True)


###############################################################################
# Evaluate one speaker
###############################################################################


def speaker(
    dataset,
    train_partition,
    test_partition,
    aggregate_metrics,
    dataset_metrics,
    checkpoint_directory,
    objective_directory,
    subjective_directory,
    index,
    adapt=promonet.ADAPTATION,
    gpu=None
):
    """Evaluate one adaptation speaker in a dataset"""
    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    if promonet.MODEL != 'world' and adapt:
        adapt_directory = checkpoint_directory / 'adapt' / dataset / index
        adapt_directory.mkdir(exist_ok=True, parents=True)

        # Maybe resume adaptation
        generator_path = torchutil.checkpoint.latest_path(
            adapt_directory,
            'generator-*.pt')
        discriminator_path = torchutil.checkpoint.latest_path(
            adapt_directory,
            'discriminator-*.pt')
        if generator_path and discriminator_path:
            checkpoint_directory = adapt_directory

        # Perform speaker adaptation
        promonet.train(
            adapt_directory,
            dataset,
            train_partition,
            test_partition,
            checkpoint_directory,
            gpu)
        checkpoint_directory = adapt_directory

    # Get generator checkpoint
    checkpoint = torchutil.checkpoint.latest_path(
        checkpoint_directory,
        'generator-*.pt')

    # Stems to use for evaluation
    test_stems = sorted(promonet.load.partition(dataset)[test_partition])
    test_stems = [stem for stem in test_stems if stem.split('/')[0] == index]
    if adapt:
        speakers = [0] * len(test_stems)
    else:
        speakers = [int(stem.split('/')[0]) for stem in test_stems]

    # Directory to save original audio files
    original_subjective_directory = \
        promonet.EVAL_DIR / 'subjective' / 'original'
    original_subjective_directory.mkdir(exist_ok=True, parents=True)

    # Directory to save original prosody files
    original_objective_directory = \
        promonet.EVAL_DIR / 'objective' / 'original'
    original_objective_directory.mkdir(exist_ok=True, parents=True)

    # Copy original files
    audio_files = []
    for stem in test_stems:
        key = f'{dataset}-{stem.replace("/", "-")}-original-100'

        # Trim to multiple of hopsize
        input_file = promonet.CACHE_DIR / dataset / f'{stem}-100.wav'
        audio = promonet.load.audio(input_file)
        trim = audio.shape[-1] % promonet.HOPSIZE
        if trim > 0:
            audio = audio[..., :-trim]

        # Save to disk
        output_file = original_subjective_directory / f'{key}.wav'
        torchaudio.save(output_file, audio, promonet.SAMPLE_RATE)
        audio_files.append(output_file)

    # Original file prefixes
    prefixes = [file.stem for file in audio_files]

    # Preprocess
    with torchutil.time.context('preprocess'):
        promonet.preprocess.from_files_to_files(
            audio_files,
            [original_objective_directory / prefix for prefix in prefixes],
            gpu=gpu,
            features=[
                'loudness',
                'pitch',
                'periodicity',
                'ppg',
                'text',
                'speaker'
            ],
            loudness_bands=None)

    ##################
    # Reconstruction #
    ##################

    original_audio_files = [
        original_subjective_directory / f'{prefix}.wav'
        for prefix in prefixes]
    files = {
        'reconstructed-100': [
            subjective_directory / f'{prefix}.wav' for prefix in prefixes]}
    viterbi = '-viterbi' if promonet.VITERBI_DECODE_PITCH else ''
    pitch_files = [
        original_objective_directory / f'{prefix}{viterbi}-pitch.pt'
        for prefix in prefixes]
    periodicity_files = [
        original_objective_directory / f'{prefix}{viterbi}-periodicity.pt'
        for prefix in prefixes]
    loudness_files = [
        original_objective_directory / f'{prefix}-loudness.pt'
        for prefix in prefixes]
    ppg_files = [
        original_objective_directory / f'{prefix}{ppgs.representation_file_extension()}'
        for prefix in prefixes]
    if promonet.ZERO_SHOT:
        speakers = [
            original_objective_directory / f'{prefix}-speaker.pt'
            for prefix in prefixes]
    if promonet.MODEL == 'world':
        synthesis_fn = functools.partial(
            promonet.baseline.world.from_files_to_files,
            pitch_files=[
                original_objective_directory /
                f'{prefix}{viterbi}-pitch.pt'
                for prefix in prefixes],
            periodicity_files=[
                original_objective_directory /
                f'{prefix}{viterbi}-periodicity.pt'
                for prefix in prefixes])
        synthesis_fn(original_audio_files, files['reconstructed-100'])
    elif promonet.SPECTROGRAM_ONLY:
        promonet.baseline.mels.from_files_to_files(
            original_audio_files,
            files['reconstructed-100'],
            checkpoint=checkpoint,
            speakers=speakers,
            gpu=gpu)
    else:
        promonet.synthesize.from_files_to_files(
            loudness_files,
            pitch_files,
            periodicity_files,
            ppg_files,
            files['reconstructed-100'],
            checkpoint=checkpoint,
            speakers=speakers,
            gpu=gpu)

    ###################
    # Prosody editing #
    ###################

    for ratio in promonet.EVALUATION_RATIOS:

        ##################
        # Pitch shifting #
        ##################

        if 'pitch' in promonet.INPUT_FEATURES:

            # Edit features
            with torchutil.time.context('edit'):
                key = f'shifted-{int(ratio * 100):03d}'
                output_prefixes = [
                    original_objective_directory /
                    prefix.replace('original-100', key)
                    for prefix in prefixes]
                promonet.edit.from_files_to_files(
                    loudness_files,
                    pitch_files,
                    periodicity_files,
                    ppg_files,
                    output_prefixes,
                    pitch_shift_cents=promonet.convert.ratio_to_cents(ratio))

            # Generate
            files[key] = [
                subjective_directory / f'{prefix.name}.wav'
                for prefix in output_prefixes]
            if promonet.MODEL == 'world':
                synthesis_fn = functools.partial(
                    promonet.baseline.world.from_files_to_files,
                    periodicity_files=[
                        f'{prefix}{viterbi}-periodicity.pt'
                        for prefix in output_prefixes])
                synthesis_fn(
                    original_audio_files,
                    files[key],
                    pitch_files=[f'{prefix}{viterbi}-pitch.pt' for prefix in output_prefixes])
            else:
                promonet.synthesize.from_files_to_files(
                    [f'{prefix}-loudness.pt' for prefix in output_prefixes],
                    [f'{prefix}{viterbi}-pitch.pt' for prefix in output_prefixes],
                    [f'{prefix}{viterbi}-periodicity.pt' for prefix in output_prefixes],
                    [f'{prefix}{ppgs.representation_file_extension()}' for prefix in output_prefixes],
                    files[key],
                    checkpoint=checkpoint,
                    speakers=speakers,
                    gpu=gpu)

        ###################
        # Time stretching #
        ###################

        if (
            'ppg' in promonet.INPUT_FEATURES and
            ppgs.REPRESENTATION_KIND == 'ppg'
        ):

            # Edit features
            with torchutil.time.context('edit'):
                key = f'stretched-{int(ratio * 100):03d}'
                output_prefixes = [
                    original_objective_directory /
                    prefix.replace('original-100', key)
                    for prefix in prefixes]
                promonet.edit.from_files_to_files(
                    loudness_files,
                    pitch_files,
                    periodicity_files,
                    ppg_files,
                    output_prefixes,
                    time_stretch_ratio=ratio,
                    stretch_unvoiced=False,
                    save_grid=True)

            # Generate
            files[key] = [
                subjective_directory / f'{prefix.name}.wav'
                for prefix in output_prefixes]
            if promonet.MODEL == 'world':
                synthesis_fn = functools.partial(
                    promonet.baseline.world.from_files_to_files,
                    pitch_files=[
                        f'{prefix}{viterbi}-pitch.pt'
                        for prefix in output_prefixes],
                    periodicity_files=[
                        f'{prefix}{viterbi}-periodicity.pt'
                        for prefix in output_prefixes])
                synthesis_fn(
                    original_audio_files,
                    files[key],
                    grid_files=[f'{prefix}-grid.pt' for prefix in output_prefixes])
            else:
                promonet.synthesize.from_files_to_files(
                    [f'{prefix}-loudness.pt' for prefix in output_prefixes],
                    [f'{prefix}{viterbi}-pitch.pt' for prefix in output_prefixes],
                    [f'{prefix}{viterbi}-periodicity.pt' for prefix in output_prefixes],
                    [f'{prefix}{ppgs.representation_file_extension()}' for prefix in output_prefixes],
                    files[key],
                    checkpoint=checkpoint,
                    speakers=speakers,
                    gpu=gpu)

        ####################
        # Loudness scaling #
        ####################

        if 'loudness' in promonet.INPUT_FEATURES:

            # Edit features
            with torchutil.time.context('edit'):
                key = f'scaled-{int(ratio * 100):03d}'
                output_prefixes = [
                    original_objective_directory /
                    prefix.replace('original-100', key)
                    for prefix in prefixes]
                promonet.edit.from_files_to_files(
                    loudness_files,
                    pitch_files,
                    periodicity_files,
                    ppg_files,
                    output_prefixes,
                    loudness_scale_db=promonet.convert.ratio_to_db(ratio))

            # Generate
            files[key] = [
                subjective_directory / f'{prefix.name}.wav'
                for prefix in output_prefixes]
            if promonet.MODEL == 'world':
                synthesis_fn = functools.partial(
                    promonet.baseline.world.from_files_to_files,
                    pitch_files=[
                        f'{prefix}{viterbi}-pitch.pt'
                        for prefix in output_prefixes],
                    periodicity_files=[
                        f'{prefix}{viterbi}-periodicity.pt'
                        for prefix in output_prefixes])
                synthesis_fn(
                    original_audio_files,
                    files[key],
                    loudness_files=[
                        f'{prefix}-loudness.pt' for prefix in output_prefixes])
            else:
                promonet.synthesize.from_files_to_files(
                    [f'{prefix}-loudness.pt' for prefix in output_prefixes],
                    [f'{prefix}{viterbi}-pitch.pt' for prefix in output_prefixes],
                    [f'{prefix}{viterbi}-periodicity.pt' for prefix in output_prefixes],
                    [f'{prefix}{ppgs.representation_file_extension()}' for prefix in output_prefixes],
                    files[key],
                    checkpoint=checkpoint,
                    speakers=speakers,
                    gpu=gpu)

        ############################
        # Spectral balance editing #
        ############################

        if promonet.AUGMENT_PITCH and promonet.MODEL != 'world':

            # Copy features
            key = f'balance-{int(ratio * 100):03d}'
            output_prefixes = [
                original_objective_directory /
                prefix.replace('original-100', key)
                for prefix in prefixes]
            for (
                loudness_file,
                pitch_file,
                periodicity_file,
                ppg_file,
                output_prefix
            ) in zip(
                loudness_files,
                pitch_files,
                periodicity_files,
                ppg_files,
                output_prefixes
            ):
                shutil.copyfile(loudness_file, f'{output_prefix}-loudness.pt')
                shutil.copyfile(
                    pitch_file,
                    f'{output_prefix}{viterbi}-pitch.pt')
                shutil.copyfile(
                    periodicity_file,
                    f'{output_prefix}{viterbi}-periodicity.pt')
                shutil.copyfile(
                    ppg_file,
                    f'{output_prefix}{ppgs.representation_file_extension()}')

            # Generate
            files[key] = [
                subjective_directory / f'{prefix.name}.wav'
                for prefix in output_prefixes]
            if promonet.SPECTROGRAM_ONLY:
                promonet.baseline.mels.from_files_to_files(
                    original_audio_files,
                    files[key],
                    speakers=speakers,
                    checkpoint=checkpoint,
                    spectral_balance_ratio=ratio,
                    gpu=gpu)
            else:
                promonet.synthesize.from_files_to_files(
                    [f'{prefix}-loudness.pt' for prefix in output_prefixes],
                    [f'{prefix}{viterbi}-pitch.pt' for prefix in output_prefixes],
                    [f'{prefix}{viterbi}-periodicity.pt' for prefix in output_prefixes],
                    [f'{prefix}{ppgs.representation_file_extension()}' for prefix in output_prefixes],
                    files[key],
                    speakers=speakers,
                    spectral_balance_ratio=ratio,
                    checkpoint=checkpoint,
                    gpu=gpu)

        ###############################
        # Perceptual loudness editing #
        ###############################

        if promonet.AUGMENT_LOUDNESS and promonet.MODEL != 'world':

            # Copy features
            key = f'loudness-{int(ratio * 100):03d}'
            output_prefixes = [
                original_objective_directory /
                prefix.replace('original-100', key)
                for prefix in prefixes]
            for (
                loudness_file,
                pitch_file,
                periodicity_file,
                ppg_file,
                output_prefix
            ) in zip(
                loudness_files,
                pitch_files,
                periodicity_files,
                ppg_files,
                output_prefixes
            ):
                shutil.copyfile(loudness_file, f'{output_prefix}-loudness.pt')
                shutil.copyfile(
                    pitch_file,
                    f'{output_prefix}{viterbi}-pitch.pt')
                shutil.copyfile(
                    periodicity_file,
                    f'{output_prefix}{viterbi}-periodicity.pt')
                shutil.copyfile(
                    ppg_file,
                    f'{output_prefix}{ppgs.representation_file_extension()}')

            # Generate
            files[key] = [
                subjective_directory / f'{prefix.name}.wav'
                for prefix in output_prefixes]
            if promonet.SPECTROGRAM_ONLY:
                promonet.baseline.mels.from_files_to_files(
                    original_audio_files,
                    files[key],
                    speakers=speakers,
                    checkpoint=checkpoint,
                    loudness_ratio=ratio,
                    gpu=gpu)
            else:
                promonet.synthesize.from_files_to_files(
                    [f'{prefix}-loudness.pt' for prefix in output_prefixes],
                    [f'{prefix}{viterbi}-pitch.pt' for prefix in output_prefixes],
                    [f'{prefix}{viterbi}-periodicity.pt' for prefix in output_prefixes],
                    [f'{prefix}{ppgs.representation_file_extension()}' for prefix in output_prefixes],
                    files[key],
                    speakers=speakers,
                    loudness_ratio=ratio,
                    checkpoint=checkpoint,
                    gpu=gpu)

    ############################
    # Speech -> representation #
    ############################

    for key, audio_files in files.items():

        # Infer speech representation
        with torchutil.time.context('preprocess'):
            promonet.preprocess.from_files_to_files(
                audio_files,
                [
                    objective_directory / file.stem
                    for file in audio_files
                ],
                gpu=gpu,
                features=[
                    'loudness',
                    'pitch',
                    'periodicity',
                    'ppg',
                    'text'
                ],
                loudness_bands=None)

    ############################
    # Evaluate prosody editing #
    ############################

    with torchutil.time.context('evaluate'):

        # Setup speaker metrics
        speaker_metrics = default_metrics()

        # Iterate over edit conditions
        results = {'objective': {'raw': {}}}
        for key, value in files.items():

            for file in value:

                # Setup file metrics
                file_metrics = promonet.evaluate.Metrics()
                if file.stem not in results['objective']['raw']:
                    results['objective']['raw'][file.stem] = {}

                # Get target filepath
                target_prefix = original_objective_directory / file.stem

                # Get predicted filepath
                predicted_prefix = objective_directory / file.stem

                # Load predicted and target features
                loudness = promonet.preprocess.loudness.band_average(
                    torch.load(f'{predicted_prefix}-loudness.pt').to(device),
                    1)
                target_loudness = promonet.preprocess.loudness.band_average(
                    torch.load(f'{target_prefix}-loudness.pt').to(device),
                    1)
                args = (
                    loudness,
                    torch.load(f'{predicted_prefix}{viterbi}-pitch.pt').to(device),
                    torch.load(f'{predicted_prefix}{viterbi}-periodicity.pt').to(device),
                    promonet.load.ppg(
                        f'{predicted_prefix}{ppgs.representation_file_extension()}',
                        loudness.shape[-1]
                    ).to(device),
                    target_loudness,
                    torch.load(f'{target_prefix}{viterbi}-pitch.pt').to(device),
                    torch.load(f'{target_prefix}{viterbi}-periodicity.pt').to(device),
                    promonet.load.ppg(
                        f'{target_prefix}{ppgs.representation_file_extension()}',
                        loudness.shape[-1]
                    ).to(device),
                    promonet.load.text(f'{predicted_prefix}.txt'),
                    promonet.load.text(
                        f'{target_prefix}.txt'.replace(key, 'original-100')))

                # Update metrics
                aggregate_metrics[key].update(*args)
                dataset_metrics[key].update(*args)
                speaker_metrics[key].update(*args)
                file_metrics.update(*args)

                # Save file results
                results['objective']['raw'][file.stem][key] = file_metrics()

    # Get results for this speaker
    results['objective']['average'] = {
        key: value() for key, value in speaker_metrics.items()}

    # Get the total number of samples we have generated
    files = subjective_directory.glob(f'{dataset}-{index}-*.wav')
    results['num_samples'] = sum(
        [torchaudio.info(file).num_frames for file in files])
    results['num_frames'] = promonet.convert.samples_to_frames(
        results['num_samples'])

    # Save to disk
    file = promonet.RESULTS_DIR / promonet.CONFIG / dataset / f'{index}.json'
    file.parent.mkdir(exist_ok=True, parents=True)
    with open(file, 'w') as file:
        json.dump(results, file, indent=4, sort_keys=True)


###############################################################################
# Utilities
###############################################################################


def default_metrics():
    """Construct the default metrics dictionary for each condition"""
    # Reconstruction metrics
    metrics = {'reconstructed-100': promonet.evaluate.Metrics()}

    # Prosody editing metrics
    if 'loudness' in promonet.INPUT_FEATURES:
        for ratio in promonet.EVALUATION_RATIOS:
            metrics[f'scaled-{int(ratio * 100):03d}'] = \
                promonet.evaluate.Metrics()
    if 'pitch' in promonet.INPUT_FEATURES:
        for ratio in promonet.EVALUATION_RATIOS:
            metrics[f'shifted-{int(ratio * 100):03d}'] = \
                promonet.evaluate.Metrics()
    if (
        'ppg' in promonet.INPUT_FEATURES and
        ppgs.REPRESENTATION_KIND == 'ppg'
    ):
        for ratio in promonet.EVALUATION_RATIOS:
            metrics[f'stretched-{int(ratio * 100):03d}'] = \
                promonet.evaluate.Metrics()

    # Spectral balance editing metrics
    if promonet.AUGMENT_PITCH and promonet.MODEL != 'world':
        for ratio in promonet.EVALUATION_RATIOS:
            metrics[f'balance-{int(ratio * 100):03d}'] = \
                promonet.evaluate.Metrics()

    # Loudness editing metrics
    if promonet.AUGMENT_LOUDNESS and promonet.MODEL != 'world':
        for ratio in promonet.EVALUATION_RATIOS:
            metrics[f'loudness-{int(ratio * 100):03d}'] = \
                promonet.evaluate.Metrics()

    return metrics
