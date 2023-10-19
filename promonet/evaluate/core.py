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
import pyfoal
import pypar
import pysodic
import torch
import torchutil
import torchaudio

import promonet


###############################################################################
# Perform evaluation
###############################################################################


@torchutil.notify.on_return('evaluate')
def datasets(datasets, checkpoint=None, gpu=None):
    """Evaluate the performance of the model on datasets"""
    aggregate_metrics = default_metrics(gpu)

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
        dataset_metrics = default_metrics(gpu)

        # Evaluate on each partition
        iterator = zip(train_partitions, test_partitions)
        for train_partition, test_partition in iterator:

            # Iterate over speakers
            indices = list(set(
                [stem.split('/')[0] for stem in partitions[test_partition]]))
            for index in indices:

                # Output directory for checkpoints and logs
                adapt_directory = (
                    promonet.RUNS_DIR /
                    promonet.CONFIG /
                    'adapt' /
                    dataset /
                    index)
                adapt_directory.mkdir(exist_ok=True, parents=True)

                # Output directory for objective evaluation
                objective_directory = (
                    promonet.EVAL_DIR /
                    'objective' /
                    promonet.CONFIG)
                (objective_directory / index).mkdir(
                    exist_ok=True,
                    parents=True)

                # Output directory for subjective evaluation
                subjective_directory = (
                    promonet.EVAL_DIR /
                    'subjective' /
                    promonet.CONFIG)
                (subjective_directory / index).mkdir(
                    exist_ok=True,
                    parents=True)

                # Evaluate a speaker
                speaker(
                    dataset,
                    train_partition,
                    test_partition,
                    checkpoint,
                    aggregate_metrics,
                    dataset_metrics,
                    adapt_directory,
                    objective_directory,
                    subjective_directory,
                    index,
                    gpu)

        # Aggregate results
        results_directory = (
            promonet.RESULTS_DIR /
            promonet.CONFIG /
            dataset)
        results = {'num_samples': 0, 'num_frames': 0}
        if promonet.MODEL != 'vits':
            results['prosody'] = {
                key: value() for key, value in dataset_metrics.items()}

        for file in results_directory.glob(f'*.json'):
            with open(file) as file:
                result = json.load(file)
            results['num_samples'] += result['num_samples']
            results['num_frames'] += result['num_frames']

        # Parse benchmarking results
        results['benchmark'] = {'raw': torchutil.time.results()}

        # Average benchmarking over samples
        results['benchmark']['average'] = {
            key: value / results['num_samples']
            for key, value in results['benchmark']['raw'].items()}

        # Print results and save to disk
        print(results)
        with open(results_directory / f'results.json', 'w') as file:
            json.dump(results, file, indent=4, sort_keys=True)

        # Plot speaker clusters
        centers = {
            index: torch.load(
                objective_directory / f'{dataset}-{index}-speaker.pt')
            for index in indices}
        embeddings = {
            index: [
                torch.load(file) for file in objective_directory.glob(
                    f'{dataset}-{index}-*-original-100-speaker.pt')]
            for index in indices}
        file = results_directory / f'speaker.pdf'
        promonet.plot.speaker.from_embeddings(centers, embeddings, file=file)


###############################################################################
# Evaluate one speaker
###############################################################################


def speaker(
    dataset,
    train_partition,
    test_partition,
    checkpoint,
    aggregate_metrics,
    dataset_metrics,
    directory,
    objective_directory,
    subjective_directory,
    index,
    gpu=None):
    """Evaluate one adaptation speaker in a dataset"""
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')

    if promonet.MODEL not in ['psola', 'world'] and promonet.ADAPTATION:

        # Maybe resume adaptation
        generator_path = torchutil.checkpoint.latest_path(
            directory,
            'generator-*.pt')
        discriminator_path = torchutil.checkpoint.latest_path(
            directory,
            'discriminator-*.pt')
        if generator_path and discriminator_path:
            checkpoint = directory

        # Perform speaker adaptation
        promonet.train(
            dataset,
            directory,
            train_partition,
            test_partition,
            checkpoint,
            gpu)

        # Get generator checkpoint
        checkpoint = torchutil.checkpoint.latest_path(
            directory,
            'generator-*.pt')

    # Stems to use for evaluation
    test_stems = sorted(promonet.load.partition(dataset)[test_partition])
    test_stems = [stem for stem in test_stems if stem.split('/')[0] == index]
    speakers = [int(stem.split('/')[0]) for stem in test_stems]

    # Directory to save original audio files
    original_subjective_directory = \
        promonet.EVAL_DIR / 'subjective' / 'original'
    (original_subjective_directory / index).mkdir(exist_ok=True, parents=True)

    # Directory to save original prosody files
    original_objective_directory = \
        promonet.EVAL_DIR / 'objective' / 'original'
    (original_objective_directory / index).mkdir(exist_ok=True, parents=True)

    # Copy original files
    for stem in test_stems:
        key = f'{dataset}-{stem.replace("/", "-")}-original-100'

        # Copy audio file
        input_file = promonet.CACHE_DIR / dataset / f'{stem}-100.wav'
        output_file = original_subjective_directory / f'{key}.wav'
        shutil.copyfile(input_file, output_file)

        # Copy text file
        input_file = promonet.CACHE_DIR / dataset / f'{stem}.txt'
        output_file = original_objective_directory / f'{key}-text.txt'
        shutil.copyfile(input_file, output_file)

        # Copy prosody files
        input_files = [
            path for path in
            (promonet.CACHE_DIR / dataset).glob(f'{stem}-100*')
            if path.suffix != '.wav']
        for input_file in input_files:
            if input_file.suffix == '.TextGrid':
                feature = 'alignment'
            else:
                feature = input_file.stem.split('-')[-1]
            output_file = (
                original_objective_directory /
                f'{key}-{feature}{input_file.suffix}')
            shutil.copyfile(input_file, output_file)

    ##################
    # Reconstruction #
    ##################

    files = {
        'original': [
            original_subjective_directory /
            f'{dataset}-{stem.replace("/", "-")}-original-100.wav'
            for stem in test_stems],
        'reconstructed': [
            subjective_directory /
            f'{dataset}-{stem.replace("/", "-")}-original-100.wav'
            for stem in test_stems]}
    pitch_files = [
        original_objective_directory /
        f'{dataset}-{stem.replace("/", "-")}-original-100-pitch.pt'
        for stem in test_stems]
    promonet.from_files_to_files(
        pitch_files,
        [
            file.parent / file.name.replace('pitch', 'periodicity')
            for file in pitch_files
        ],
        [
            file.parent / file.name.replace('pitch', 'loudness')
            for file in pitch_files
        ],
        [
            file.parent / file.name.replace('pitch', 'ppg')
            for file in pitch_files
        ],
        files['reconstructed'],
        checkpoint=checkpoint,
        speakers=(
            [0] * len(test_stems) if promonet.ADAPTATION else speakers
        ),
        gpu=gpu)

    # Perform speech editing only on speech editors
    if promonet.MODEL in ['hifigan', 'vits']:

        results = {}

    else:

        ##################
        # Pitch shifting #
        ##################

        for ratio in promonet.EVALUATION_RATIOS:
            key = f'shifted-{int(ratio * 100):03d}'
            for stem in test_stems:
                prefix = f'{dataset}-{stem.replace("/", "-")}'
                file = (
                    original_objective_directory /
                    f'{prefix}-original-100-pitch.pt')

                # Shift original pitch and save to disk
                output_prefix = \
                    original_objective_directory / f'{prefix}-{key}'
                output_prefix.parent.mkdir(exist_ok=True, parents=True)
                promonet.edit.from_file_to_file(
                    file,
                    file.parent / file.name.replace('pitch', 'periodicity'),
                    file.parent / file.name.replace('pitch', 'loudness'),
                    file.parent / file.name.replace('pitch', 'ppg'),
                    output_prefix,
                    pitch_shift_cents=promonet.convert.ratio_to_cents(ratio))

                # Copy unchanged evaluation features
                for feature in ['phonemes', 'text', 'voicing']:
                    suffix = '.txt' if feature == 'text' else '.pt'
                    input_file = (
                        original_objective_directory /
                        f'{prefix}-original-100-{feature}{suffix}')
                    output_file = f'{output_prefix}-{feature}{suffix}'
                    shutil.copyfile(input_file, output_file)

            # Generate
            files[key] = [
                subjective_directory /
                f'{dataset}-{stem.replace("/", "-")}-{key}.wav'
                for stem in test_stems]
            pitch_files = [
                original_objective_directory /
                f'{dataset}-{stem.replace("/", "-")}-{key}-pitch.pt'
                for stem in test_stems]
            promonet.from_files_to_files(
                pitch_files,
                [
                    file.parent / file.name.replace('pitch', 'periodicity')
                    for file in pitch_files
                ],
                [
                    file.parent / file.name.replace('pitch', 'loudness')
                    for file in pitch_files
                ],
                [
                    file.parent / file.name.replace('pitch', 'ppg')
                    for file in pitch_files
                ],
                files[key],
                checkpoint=checkpoint,
                speakers=(
                    [0] * len(test_stems) if promonet.ADAPTATION else speakers
                ),
                gpu=gpu)

        ###################
        # Time stretching #
        ###################

        for ratio in promonet.EVALUATION_RATIOS:
            key = f'stretched-{int(ratio * 100):03d}'

            # Stretch original alignment and save to disk
            for stem in test_stems:
                prefix = f'{dataset}-{stem.replace("/", "-")}'
                file = (
                    original_objective_directory /
                    f'{prefix}-original-100-alignment.TextGrid')

                # Load alignment
                alignment = pypar.Alignment(file)

                # Interpolate voiced regions
                # TODO - can we do this with PPGs instead of forced alignment?
                interpolated = pyfoal.interpolate.voiced(alignment, ratio)
                grid = promonet.interpolate.grid.from_alignments(
                    alignment,
                    interpolated)

                # Save alignment to disk
                alignment_file = (
                    original_objective_directory /
                    f'{prefix}-{key}-alignment.TextGrid')
                alignment_file.parent.mkdir(exist_ok=True, parents=True)
                interpolated.save(alignment_file)

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
                        original_objective_directory /
                        f'{prefix}-original-100-{feature}.pt')
                    output_file = (
                        original_objective_directory /
                        f'{prefix}-{key}-{feature}.pt')
                    original_feature = torch.load(input_file)
                    if size is None:
                        size = original_feature.shape[-1]
                    if feature in ['loudness', 'periodicity']:
                        stretched_feature = promonet.interpolate.grid_sample(
                            original_feature.squeeze(),
                            grid.squeeze(),
                            'linear')[None]
                    elif feature in ['phonemes', 'voicing']:
                        stretched_feature = promonet.interpolate.grid_sample(
                            original_feature.squeeze(),
                            grid.squeeze(),
                            'nearest')[None]
                    elif feature == 'ppg':
                        original_feature = promonet.interpolate.ppg(
                            original_feature.to(torch.float32),
                            promonet.interpolate.grid.of_length(
                                original_feature,
                                size))
                        stretched_feature = promonet.interpolate.ppg(
                            original_feature.squeeze(),
                            grid.squeeze())
                    elif feature == 'pitch':
                        stretched_feature = promonet.interpolate.pitch(
                            original_feature.squeeze(),
                            grid.squeeze())[None]
                    torch.save(stretched_feature, output_file)

                # Copy text
                input_file = (
                    original_objective_directory /
                    f'{prefix}-original-100-text.txt')
                output_file = \
                    original_objective_directory / f'{prefix}-{key}-text.txt'
                shutil.copyfile(input_file, output_file)

            # Generate
            files[key] = [
                subjective_directory /
                f'{dataset}-{stem.replace("/", "-")}-{key}.wav'
                for stem in test_stems]
            pitch_files = [
                original_objective_directory /
                f'{dataset}-{stem.replace("/", "-")}-{key}-pitch.pt'
                for stem in test_stems]
            promonet.from_files_to_files(
                pitch_files,
                [
                    file.parent / file.name.replace('pitch', 'periodicity')
                    for file in pitch_files
                ],
                [
                    file.parent / file.name.replace('pitch', 'loudness')
                    for file in pitch_files
                ],
                [
                    file.parent / file.name.replace('pitch', 'ppg')
                    for file in pitch_files
                ],
                files[key],
                checkpoint=checkpoint,
                speakers=(
                    [0] * len(test_stems) if promonet.ADAPTATION else speakers
                ),
                gpu=gpu)

        ####################
        # Loudness scaling #
        ####################

        for ratio in promonet.EVALUATION_RATIOS:
            key = f'scaled-{int(ratio * 100):03d}'
            for stem in test_stems:
                prefix = f'{dataset}-{stem.replace("/", "-")}'
                file = (
                    original_objective_directory /
                    f'{prefix}-original-100-loudness.pt')

                # Scale original loudness and save to disk
                output_prefix = (
                    original_objective_directory /
                    f'{prefix}-{key}')
                promonet.edit.from_file_to_file(
                    file.parent / file.name.replace('loudness', 'pitch'),
                    file.parent / file.name.replace('loudness', 'periodicity'),
                    file,
                    file.parent / file.name.replace('loudness', 'ppg'),
                    output_prefix,
                    loudness_scale_db=promonet.convert.ratio_to_db(ratio))

                # Copy unchanged evaluation features
                for feature in ['phonemes', 'text', 'voicing']:
                    suffix = '.txt' if feature == 'text' else '.pt'
                    input_file = (
                        original_objective_directory /
                        f'{prefix}-original-100-{feature}{suffix}')
                    output_file = f'{output_prefix}-{feature}{suffix}'
                    shutil.copyfile(input_file, output_file)

            # Generate
            files[key] = [
                subjective_directory /
                f'{dataset}-{stem.replace("/", "-")}-{key}.wav'
                for stem in test_stems]
            pitch_files = [
                original_objective_directory /
                f'{dataset}-{stem.replace("/", "-")}-{key}-pitch.pt'
                for stem in test_stems]
            promonet.from_files_to_files(
                pitch_files,
                [
                    file.parent / file.name.replace('pitch', 'periodicity')
                    for file in pitch_files
                ],
                [
                    file.parent / file.name.replace('pitch', 'loudness')
                    for file in pitch_files
                ],
                [
                    file.parent / file.name.replace('pitch', 'ppg')
                    for file in pitch_files
                ],
                files[key],
                checkpoint=checkpoint,
                speakers=(
                    [0] * len(test_stems) if promonet.ADAPTATION else speakers
                ),
                gpu=gpu)

    ############################
    # Speech -> representation #
    ############################

    for audio_files in files.values():

        # Infer PPGs
        ppg_files = [
            objective_directory / f'{file.stem}-ppg.pt'
            for file in audio_files]
        ppgs.from_files_to_files(audio_files, ppg_files, gpu=gpu)

        # Infer prosody
        text_files = [
            original_objective_directory / f'{file.stem}-text.txt'
            for file in audio_files]
        prefixes = [objective_directory / file.stem for file in audio_files]
        pysodic.from_files_to_files(
            audio_files,
            prefixes,
            text_files,
            promonet.HOPSIZE / promonet.SAMPLE_RATE,
            promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
            gpu=gpu)

        # Infer speaker embeddings
        embedding_files = [
            objective_directory / f'{file.stem}-speaker.pt'
            for file in audio_files]
        promonet.resemblyzer.from_files_to_files(
            audio_files,
            embedding_files,
            gpu=gpu)
    original_files = original_subjective_directory.glob(
        f'{dataset}-{index}-*-original-100.wav')
    speaker_embedding = promonet.resemblyzer.from_files(
        original_files,
        gpu)
    torch.save(
        speaker_embedding,
        objective_directory / f'{dataset}-{index}-speaker.pt')

    ############################
    # Evaluate prosody editing #
    ############################

    if promonet.MODEL != 'vits':

        # Setup speaker metrics
        speaker_metrics = default_metrics(gpu)

        # Iterate over edit conditions
        results = {'objective': {'raw': {}}}
        for key, value in files.items():
            results['objective']['raw'][key] = []
            for file in value:

                # Get prosody metrics
                file_metrics = promonet.evaluate.Metrics(gpu)

                # Get target filepath
                target_prefix = original_objective_directory / file.stem

                # Get predicted filepath
                predicted_prefix = objective_directory / file.stem

                # Update metrics
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

                # Get predicted and target PPGs
                grid = promonet.interpolate.grid.of_length(
                    prosody_args[0].shape[-1])
                predicted_ppgs = promonet.interpolate.ppg(
                    torch.load(f'{predicted_prefix}-ppg.pt').to(device),
                    grid)
                target_ppgs = promonet.interpolate.ppg(
                    torch.load(f'{target_prefix}-ppg.pt').to(device),
                    grid)
                ppg_args = (predicted_ppgs, target_ppgs)

                # Get target text and audio for WER
                text = promonet.load.text(f'{target_prefix}-text.txt')
                audio = promonet.load.audio(file)
                wer_args = (text, audio)

                # Get speaker embeddings
                embedding = torch.load(f'{predicted_prefix}-speaker.pt').to(
                    speaker_embedding.device)
                speaker_sim_args = (speaker_embedding, embedding)

                # Update metrics
                condition = '-'.join(target_prefix.stem.split('-')[3:5])
                args = (prosody_args, ppg_args, wer_args, speaker_sim_args)
                aggregate_metrics[condition].update(*args)
                dataset_metrics[condition].update(*args)
                speaker_metrics[condition].update(*args)
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

    # Print results and save to disk
    print(results)
    file = (
        promonet.RESULTS_DIR /
        promonet.CONFIG /
        dataset /
        f'{index}.json')
    file.parent.mkdir(exist_ok=True, parents=True)
    with open(file, 'w') as file:
        json.dump(results, file, indent=4, sort_keys=True)


###############################################################################
# Utilities
###############################################################################


def default_metrics(gpu):
    """Construct the default metrics dictionary for each condition"""
    # Bind shared parameter
    metric_fn = functools.partial(promonet.evaluate.Metrics, gpu)

    # Reconstruction metrics
    metrics = {'original-100': metric_fn()}

    if promonet.MODEL not in ['hifigan', 'vits']:

        # Prosody editing metrics
        for condition in ['scaled', 'shifted', 'stretched']:
            for ratio in promonet.EVALUATION_RATIOS:
                metrics[f'{condition}-{int(ratio * 100):03d}'] = metric_fn()

    return metrics
