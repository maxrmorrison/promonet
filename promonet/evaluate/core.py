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
import json

import ppgs
import torch
import torchaudio
import torchutil

import promonet


###############################################################################
# Perform evaluation
###############################################################################


@torchutil.notify('evaluate')
def datasets(datasets, gpu=None):
    """Evaluate the performance of the model on datasets"""
    aggregate_metrics = default_metrics()

    # Evaluate on each dataset
    for dataset in datasets:

        # Reset benchmarking
        torchutil.time.reset()

        # Get adaptation partitions for this dataset
        partitions = promonet.load.partition(dataset)
        if promonet.ADAPTATION:
            train_partitions = sorted(list(
                partition for partition in partitions.keys()
                if 'train-adapt' in partition))
            test_partitions = sorted(list(
                partition for partition in partitions.keys()
                if 'test-adapt' in partition))
        else:
            train_partitions = [None]
            test_partitions = ['test']

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
                    gpu)

        # Aggregate results
        results_directory = promonet.RESULTS_DIR / promonet.CONFIG / dataset
        results_directory.mkdir(exist_ok=True, parents=True)
        results = {'num_samples': 0, 'num_frames': 0}
        if promonet.MODEL != 'vits':
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
            key: value / (results['num_samples'] / promonet.SAMPLE_RATE)
            for key, value in results['benchmark']['raw'].items()}

        # Print results and save to disk
        with open(results_directory / f'results.json', 'w') as file:
            json.dump(results, file, indent=4, sort_keys=True)

        # Plot speaker clusters
        # centers = {
        #     index: torch.load(
        #         objective_directory / f'{dataset}-{index}-speaker.pt')
        #     for index in indices}
        # embeddings = {
        #     index: [
        #         torch.load(file) for file in objective_directory.glob(
        #             f'{dataset}-{index}-*-original-100-speaker.pt')]
        #     for index in indices}
        # file = results_directory / f'speaker.pdf'
        # promonet.plot.speaker.from_embeddings(centers, embeddings, file=file)

    # Get aggregate metrics
    results = {key: value() for key, value in aggregate_metrics.items()}

    # Compute aggregate benchmark
    results |= {'num_samples': 0, 'num_frames': 0}
    results_directory = promonet.RESULTS_DIR / promonet.CONFIG
    for file in results_directory.glob(f'*/results.json'):
        with open(file) as file:
            result = json.load(file)
        results['num_samples'] += result['num_samples']
        results['num_frames'] += result['num_frames']
        if 'benchmark' not in results:
            results['benchmark'] = {'raw': result['benchmark']['raw']}
        else:
            for key, value in result['benchmark']['raw']:
                results['benchmark']['raw']['key'] += value
    results['benchmark']['rtf'] = {
        key: value / (results['num_samples'] / promonet.SAMPLE_RATE)
        for key, value in results['benchmark']['raw'].items()}

    # Print results and save to disk
    print(results)
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
    gpu=None
):
    """Evaluate one adaptation speaker in a dataset"""
    device = f'cuda:{gpu}' if gpu is not None else 'cpu'

    if promonet.MODEL not in ['psola', 'world'] and promonet.ADAPTATION:
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
    if promonet.ADAPTATION:
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
            features=['loudness', 'periodicity', 'pitch', 'ppg', 'text'])

    ##################
    # Reconstruction #
    ##################

    original_audio_files = [
        original_subjective_directory / f'{prefix}.wav'
        for prefix in prefixes]
    files = {
        'reconstructed-100': [
            subjective_directory / f'{prefix}.wav' for prefix in prefixes]}
    pitch_files = [
        original_objective_directory / f'{prefix}-pitch.pt'
        for prefix in prefixes]
    periodicity_files = [
        original_objective_directory / f'{prefix}-periodicity.pt'
        for prefix in prefixes]
    loudness_files = [
        original_objective_directory / f'{prefix}-loudness.pt'
        for prefix in prefixes]
    ppg_files = [
        original_objective_directory / f'{prefix}{ppgs.representation_file_extension()}'
        for prefix in prefixes]
    if promonet.MODEL in ['psola', 'world']:
        if promonet.MODEL == 'psola':
            synthesis_fn = promonet.baseline.psola.from_files_to_files
        elif promonet.MODEL == 'world':
            synthesis_fn = promonet.baseline.world.from_files_to_files
        synthesis_fn(original_audio_files, files['reconstructed-100'])
    else:
        promonet.synthesize.from_files_to_files(
            pitch_files,
            periodicity_files,
            loudness_files,
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
                    pitch_files,
                    periodicity_files,
                    loudness_files,
                    ppg_files,
                    output_prefixes,
                    pitch_shift_cents=promonet.convert.ratio_to_cents(ratio))

            # Generate
            files[key] = [
                subjective_directory / f'{prefix.name}.wav'
                for prefix in output_prefixes]
            if promonet.MODEL in ['psola', 'world']:
                if promonet.MODEL == 'psola':
                    synthesis_fn = promonet.baseline.psola.from_files_to_files
                elif promonet.MODEL == 'world':
                    synthesis_fn = promonet.baseline.world.from_files_to_files
                synthesis_fn(
                    original_audio_files,
                    files[key],
                    pitch_files=[f'{prefix}-pitch.pt' for prefix in output_prefixes])
            else:
                promonet.synthesize.from_files_to_files(
                    [f'{prefix}-pitch.pt' for prefix in output_prefixes],
                    [f'{prefix}-periodicity.pt' for prefix in output_prefixes],
                    [f'{prefix}-loudness.pt' for prefix in output_prefixes],
                    [f'{prefix}{ppgs.representation_file_extension()}' for prefix in output_prefixes],
                    files[key],
                    checkpoint=checkpoint,
                    speakers=speakers,
                    gpu=gpu)

        ###################
        # Time stretching #
        ###################

        if 'ppg' in promonet.INPUT_FEATURES:

            # Edit features
            with torchutil.time.context('edit'):
                key = f'stretched-{int(ratio * 100):03d}'
                output_prefixes = [
                    original_objective_directory /
                    prefix.replace('original-100', key)
                    for prefix in prefixes]
                promonet.edit.from_files_to_files(
                    pitch_files,
                    periodicity_files,
                    loudness_files,
                    ppg_files,
                    output_prefixes,
                    time_stretch_ratio=ratio,
                    stretch_unvoiced=False,
                    save_grid=True)

            # Generate
            files[key] = [
                subjective_directory / f'{prefix.name}.wav'
                for prefix in output_prefixes]
            if promonet.MODEL in ['psola', 'world']:
                if promonet.MODEL == 'psola':
                    synthesis_fn = promonet.baseline.psola.from_files_to_files
                elif promonet.MODEL == 'world':
                    synthesis_fn = promonet.baseline.world.from_files_to_files
                synthesis_fn(
                    original_audio_files,
                    files[key],
                    grid_files=[f'{prefix}-grid.pt' for prefix in output_prefixes])
            else:
                promonet.synthesize.from_files_to_files(
                    [f'{prefix}-pitch.pt' for prefix in output_prefixes],
                    [f'{prefix}-periodicity.pt' for prefix in output_prefixes],
                    [f'{prefix}-loudness.pt' for prefix in output_prefixes],
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
                    pitch_files,
                    periodicity_files,
                    loudness_files,
                    ppg_files,
                    output_prefixes,
                    loudness_scale_db=promonet.convert.ratio_to_db(ratio))

            # Generate
            files[key] = [
                subjective_directory / f'{prefix.name}.wav'
                for prefix in output_prefixes]
            if promonet.MODEL in ['psola', 'world']:
                if promonet.MODEL == 'psola':
                    synthesis_fn = promonet.baseline.psola.from_files_to_files
                elif promonet.MODEL == 'world':
                    synthesis_fn = promonet.baseline.world.from_files_to_files
                synthesis_fn(
                    original_audio_files,
                    files[key],
                    loudness_files=[
                        f'{prefix}-loudness.pt' for prefix in output_prefixes])
            else:
                promonet.synthesize.from_files_to_files(
                    [f'{prefix}-pitch.pt' for prefix in output_prefixes],
                    [f'{prefix}-periodicity.pt' for prefix in output_prefixes],
                    [f'{prefix}-loudness.pt' for prefix in output_prefixes],
                    [f'{prefix}{ppgs.representation_file_extension()}' for prefix in output_prefixes],
                    files[key],
                    checkpoint=checkpoint,
                    speakers=speakers,
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
                features=['loudness', 'periodicity', 'pitch', 'ppg', 'text'])

        # Infer speaker embeddings
        # embedding_files = [
        #     objective_directory / f'{file.stem}-speaker.pt'
        #     for file in audio_files]
        # promonet.resemblyzer.from_files_to_files(
        #     audio_files,
        #     embedding_files,
        #     gpu=gpu)

    # original_files = original_subjective_directory.glob(
    #     f'{dataset}-{index}-*-original-100.wav')
    # speaker_embedding = promonet.resemblyzer.from_files(original_files, gpu)
    # torch.save(
    #     speaker_embedding,
    #     objective_directory / f'{dataset}-{index}-speaker.pt')

    ############################
    # Evaluate prosody editing #
    ############################

    with torchutil.time.context('evaluate'):

        # Setup speaker metrics
        speaker_metrics = default_metrics()

        # Iterate over edit conditions
        results = {'objective': {'raw': {}}}
        for key, value in files.items():

            results['objective']['raw'][key] = []
            for file in value:

                # Get prosody metrics
                file_metrics = promonet.evaluate.Metrics()

                # Get target filepath
                target_prefix = original_objective_directory / file.stem

                # Get predicted filepath
                predicted_prefix = objective_directory / file.stem

                # Load predicted and target features
                pitch = torch.load(f'{predicted_prefix}-pitch.pt').to(device)
                args = (
                    pitch,
                    torch.load(f'{predicted_prefix}-periodicity.pt').to(device),
                    torch.load(f'{predicted_prefix}-loudness.pt').to(device),
                    promonet.load.ppg(f'{predicted_prefix}{ppgs.representation_file_extension()}', pitch.shape[-1]).to(device),
                    torch.load(f'{target_prefix}-pitch.pt').to(device),
                    torch.load(f'{target_prefix}-periodicity.pt').to(device),
                    torch.load(f'{target_prefix}-loudness.pt').to(device),
                    promonet.load.ppg(f'{target_prefix}{ppgs.representation_file_extension()}', pitch.shape[-1]).to(device),
                    promonet.load.text(f'{predicted_prefix}.txt'),
                    promonet.load.text(f'{target_prefix}.txt'.replace(key, 'original-100')),
                    None)

                # Get speaker embeddings
                # embedding = torch.load(f'{predicted_prefix}-speaker.pt').to(
                #     speaker_embedding.device)
                # speaker_sim_args = (speaker_embedding, embedding)

                # Update metrics
                aggregate_metrics[key].update(*args)
                dataset_metrics[key].update(*args)
                speaker_metrics[key].update(*args)
                file_metrics.update(*args)

            # Get results for this file
            results['objective']['raw'][key].append(
                (file.stem, file_metrics()))

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
    if 'ppg' in promonet.INPUT_FEATURES:
        for ratio in promonet.EVALUATION_RATIOS:
            metrics[f'stretched-{int(ratio * 100):03d}'] = \
                promonet.evaluate.Metrics()

    return metrics
