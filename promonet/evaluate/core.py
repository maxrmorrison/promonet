"""Objective and subjective evaluation of prosody editing

Files generated during evaluation are saved in eval/. The directory structure
is as follows.

eval
└── <condition>
    ├── objective
    |    └── <dataset>
    |        └── <speaker>
    |            └── <utterance>-<modification>-<ratio>-<feature>.<extension>
    └── subjective
        └── <dataset>
            └── <speaker>
                └── <utterance>-<modification>-<ratio>.wav
"""
import functools
import json
import shutil

import pyfoal
import pypar
import pysodic
import torch
import torchaudio
import ppgs
import resemblyzer
import numpy as np
import matplotlib.pyplot as plt

import promonet


###############################################################################
# Constants
###############################################################################


# Constant ratios at which we evaluate prosody
if promonet.PITCH_EVAL_METHOD == 'ratio':
    RATIOS = [.717, 1.414]
else:
    RATIOS = [2 ** (cents / 1200) for cents in promonet.PITCH_CENTS]



###############################################################################
# Perform evaluation
###############################################################################


def datasets(datasets, checkpoint=None, gpus=None):
    """Evaluate the performance of the model on datasets"""
    # Turn on benchmarking
    current_benchmark = promonet.BENCHMARK
    promonet.BENCHMARK = True

    print("ratios:", RATIOS, flush=True)

    # Evaluate on each dataset
    for dataset in datasets:

        # Reset benchmarking
        promonet.TIMER.reset()

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

        # Prosody metrics
        metrics = default_metrics(gpus)

        # Evaluate on each partition
        iterator = zip(train_partitions, test_partitions)
        for train_partition, test_partition in iterator:

            # Get speaker index
            # index = partitions[test_partition][0].split('/')[0]

            indices = list(set([stem.split('/')[0] for stem in partitions[test_partition]]))
            print('speakers:', indices)
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
                    promonet.CONFIG /
                    'objective' /
                    dataset)
                (objective_directory / index).mkdir(exist_ok=True, parents=True)

                # Output directory for subjective evaluation
                subjective_directory = (
                    promonet.EVAL_DIR /
                    promonet.CONFIG /
                    'subjective' /
                    dataset)
                (subjective_directory / index).mkdir(exist_ok=True, parents=True)

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
                    index,
                    gpus)

        # Aggregate results
        results_directory = (
            promonet.EVAL_DIR /
            promonet.CONFIG /
            'objective' /
            dataset)
        results = {'num_samples': 0, 'num_frames': 0}
        if promonet.MODEL != 'vits':
            results['prosody'] = {key: value()
                                  for key, value in metrics.items()}

        print(list(results_directory.rglob('*/results.json')))
        for file in results_directory.rglob('*/results.json'):
            with open(file) as file:
                result = json.load(file)
            results['num_samples'] += result['num_samples']
            results['num_frames'] += result['num_frames']

        # Parse benchmarking results
        results['benchmark'] = {'raw': promonet.TIMER()}

        # Average benchmarking over samples
        results['benchmark']['average'] = {
            key: value / results['num_samples']
            for key, value in results['benchmark']['raw'].items()}

        # Print results and save to disk
        print(results)
        with open(results_directory / 'results.json', 'w') as file:
            json.dump(results, file, indent=4, sort_keys=True)

    # Maybe turn off benchmarking
    promonet.BENCHMARK = current_benchmark


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
    index,
    gpus=None):
    """Evaluate one adaptation speaker in a dataset"""

    device = torch.device(f'cuda:{gpus[0]}' if gpus is not None else 'cpu')

    plot_dir = promonet.PLOT_DIR / promonet.CONFIG
    plot_dir.mkdir(exist_ok=True, parents=True)

    if promonet.MODEL not in ['psola', 'world'] and promonet.ADAPTATION:

        # Maybe resume adaptation
        generator_path = promonet.checkpoint.latest_path(
            output_directory,
            'generator-*.pt')
        discriminator_path = promonet.checkpoint.latest_path(
            output_directory,
            'discriminator-*.pt')
        if generator_path and discriminator_path:
            checkpoint = output_directory

        # Perform speaker adaptation and get generator checkpoint
        checkpoint = promonet.train.run(
            dataset,
            checkpoint,
            output_directory,
            log_directory,
            train_partition,
            test_partition,
            True,
            gpus)

    # Stems to use for evaluation
    test_stems = sorted(promonet.load.partition(dataset)[test_partition])
    test_stems = [stem for stem in test_stems if stem.split('/')[0] == index]
    speaker_ids = [int(stem.split('/')[0]) for stem in test_stems]
    print(test_stems, speaker_ids)

    # Directory to save original audio files
    original_subjective_directory = \
        promonet.EVAL_DIR / 'original' / 'subjective' / dataset
    (original_subjective_directory / index).mkdir(exist_ok=True, parents=True)

    # Directory to save original prosody files
    original_objective_directory = \
        promonet.EVAL_DIR / 'original' / 'objective' / dataset
    (original_objective_directory / index).mkdir(exist_ok=True, parents=True)

    # Evaluation device
    gpu = None if gpus is None else gpus[0]

    # Copy original files
    for stem in test_stems:

        # Copy audio file
        input_file = promonet.CACHE_DIR / dataset / f'{stem}-100.wav'
        output_file = \
            original_subjective_directory / f'{stem}-original-100.wav'
        shutil.copyfile(input_file, output_file)

        # Copy text file
        input_file = promonet.CACHE_DIR / dataset / f'{stem}.txt'
        output_file = \
            original_objective_directory / f'{stem}-original-100-text.txt'
        shutil.copyfile(input_file, output_file)

        # Copy prosody files
        input_files = [
            path for path in (promonet.CACHE_DIR / dataset).glob(f'{stem}-100*')
            if path.suffix != '.wav']
        for input_file in input_files:
            if input_file.suffix == '.TextGrid':
                feature = 'alignment'
            elif promonet.PPG_MODEL is not None and str(input_file).endswith(f'{promonet.PPG_MODEL}.pt'):
                feature = promonet.PPG_MODEL
            else:
                feature = input_file.stem.split('-')[-1]
            output_file = (
                original_objective_directory /
                f'{stem}-original-100-{feature}{input_file.suffix}')
            shutil.copyfile(input_file, output_file)

    ##################
    # Reconstruction #
    ##################

    files = {
        'original': sorted([
            original_subjective_directory / f'{stem}-original-100.wav'
            for stem in test_stems])}
    files['reconstructed'] = sorted([
        subjective_directory / f'{stem}-original-100.wav'
        for stem in test_stems])
    text_files = [
        original_objective_directory /
        file.parent.name /
        f'{file.stem}-text.txt'
        for file in files['original']]
    promonet.from_files_to_files(
        files['original'],
        files['reconstructed'],
        text_files=text_files,
        checkpoint=checkpoint,
        speaker_ids=speaker_ids,
        gpu=None if promonet.MODEL in ['psola', 'world'] else gpu)

    # Copy unchanged prosody features
    for file in files['reconstructed']:
        features = [
            'loudness',
            'periodicity',
            'phonemes',
            'pitch',
            'ppg' if promonet.PPG_MODEL is None else promonet.PPG_MODEL,
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

    # Perform speech editing only on speech editors
    if promonet.MODEL in ['hifigan', 'vits']:

        results = {}

    else:

        ##################
        # Pitch shifting #
        ##################

        original_pitch_files = sorted([
            original_objective_directory / f'{stem}-original-100-pitch.pt'
            for stem in test_stems])
        for ratio in RATIOS:

            # Shift original pitch and save to disk
            for original_pitch_file in original_pitch_files:
                stem = original_pitch_file.stem[:6]
                key = f'shifted-{int(ratio * 100):03d}'
                pitch = ratio * torch.load(original_pitch_file)
                pitch[pitch < promonet.FMIN] = promonet.FMIN
                pitch[pitch > promonet.FMAX] = promonet.FMAX
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
                    'ppg' if promonet.PPG_MODEL is None else promonet.PPG_MODEL,
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
            files[key] = sorted([
                subjective_directory / f'{stem}-{key}.wav'
                for stem in test_stems])
            pitch_files = sorted([
                original_objective_directory / f'{stem}-{key}-pitch.pt'
                for stem in test_stems])

            # Generate
            promonet.from_files_to_files(
                files['original'],
                files[key],
                target_pitch_files=pitch_files,
                checkpoint=checkpoint,
                speaker_ids=speaker_ids,
                gpu=None if promonet.MODEL in ['psola', 'world'] else gpu)

        ###################
        # Time stretching #
        ###################

        original_alignment_files = sorted([
            original_objective_directory /
            f'{stem}-original-100-alignment.TextGrid'
            for stem in test_stems])
        for ratio in RATIOS:
            key = f'stretched-{int(ratio * 100):03d}'

            # Stretch original alignment and save to disk
            for original_alignment_file in original_alignment_files:

                # Load alignment
                alignment = pypar.Alignment(original_alignment_file)

                # Interpolate voiced regions
                interpolated = pyfoal.interpolate.voiced(alignment, ratio)
                grid = promonet.interpolate.grid.from_alignments(
                    alignment,
                    interpolated)

                # Save alignment to disk
                alignment_file = (
                    original_objective_directory /
                    original_alignment_file.parent.name /
                    f'{original_alignment_file.stem[:6]}-{key}-alignment.TextGrid')
                alignment_file.parent.mkdir(exist_ok=True, parents=True)
                interpolated.save(alignment_file)

                # Stretch and save other prosody features
                features = [
                    'loudness',
                    'periodicity',
                    'phonemes',
                    'pitch',
                    'ppg' if promonet.PPG_MODEL is None else promonet.PPG_MODEL,
                    'voicing']
                size = None
                for feature in features:
                    input_file = (
                        alignment_file.parent /
                        alignment_file.name.replace(
                            key, 'original-100').replace('alignment', feature)).with_suffix('.pt')
                    output_file = (
                        alignment_file.parent /
                        alignment_file.name.replace('alignment', feature)).with_suffix('.pt')
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
                    elif feature == 'ppg' or feature == promonet.PPG_MODEL:
                        # if ppgs.FRONTEND is not None and promonet.PPG_MODEL is not None:
                            # if not hasattr(speaker, 'frontend'):
                            #     speaker.frontend = ppgs.FRONTEND(original_feature.device)
                            # original_feature = speaker.frontend(original_feature.unsqueeze(dim=0)).squeeze(dim=0)
                        original_feature = original_feature.to(torch.float32)
                        mode = promonet.PPG_INTERP_METHOD
                        original_feature = torch.nn.functional.interpolate(
                            original_feature[None].to(torch.float32),
                            size=size,
                            mode=mode,
                            align_corners=None if mode == 'nearest' else False
                        )[0]
                        stretched_feature = promonet.interpolate.ppgs(
                            original_feature.squeeze(),
                            grid.squeeze())
                    elif feature == 'pitch':
                        stretched_feature = promonet.interpolate.pitch(
                            original_feature.squeeze(),
                            grid.squeeze())[None]
                    torch.save(stretched_feature, output_file)

                # Copy text
                input_file = (
                    alignment_file.parent /
                    alignment_file.name.replace(
                        key, 'original-100').replace(
                        'alignment', 'text')).with_suffix('.txt')
                output_file = (
                    alignment_file.parent /
                    alignment_file.name.replace('alignment', 'text')).with_suffix('.txt')
                shutil.copyfile(input_file, output_file)

            # Get filenames
            files[key] = sorted([
                subjective_directory / f'{stem}-{key}.wav'
                for stem in test_stems])
            alignment_files = sorted([
                original_objective_directory / f'{stem}-{key}-alignment.TextGrid'
                for stem in test_stems])
            text_files = sorted([
                original_objective_directory / f'{stem}-{key}-text.txt'
                for stem in test_stems])
            # Generate
            promonet.from_files_to_files(
                files['original'],
                files[key],
                text_files=text_files,
                alignment_files=alignment_files,
                checkpoint=checkpoint,
                speaker_ids=speaker_ids,
                gpu=None if promonet.MODEL in ['psola', 'world'] else gpu)

        ####################
        # Loudness scaling #
        ####################

        original_loudness_files = sorted([
            original_objective_directory / f'{stem}-original-100-loudness.pt'
            for stem in test_stems])
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
                    'ppg' if promonet.PPG_MODEL is None else promonet.PPG_MODEL,
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
            files[key] = sorted([
                subjective_directory / f'{stem}-{key}.wav'
                for stem in test_stems])
            loudness_files = sorted([
                original_objective_directory / f'{stem}-{key}-loudness.pt'
                for stem in test_stems])

            # Generate
            promonet.from_files_to_files(
                files['original'],
                files[key],
                target_loudness_files=loudness_files,
                checkpoint=checkpoint,
                speaker_ids=speaker_ids,
                gpu=None if promonet.MODEL in ['psola', 'world'] else gpu)

    ############################
    # Speech -> representation #
    ############################

    for audio_files in files.values():
        # Preprocess phonetic posteriorgrams
        ppg_files = [
            objective_directory / file.parent.name / f'{file.stem}-{promonet.PPG_MODEL if promonet.PPG_MODEL is not None else "ppg"}.pt'
            for file in audio_files]
        promonet.data.preprocess.ppg.from_files_to_files(
            audio_files,
            ppg_files,
            gpu)

        # Preprocess prosody features
        text_files = [
            original_objective_directory /
            file.parent.name /
            f'{file.stem}-text.txt'
            for file in audio_files]
        prefixes = [
            objective_directory / file.parent.name / file.stem
            for file in audio_files]
        pysodic.from_files_to_files(
            audio_files,
            prefixes,
            text_files,
            promonet.HOPSIZE / promonet.SAMPLE_RATE,
            promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
            gpu=gpu)

    if promonet.MODEL != 'vits':

        ############
        # Evaluate #
        ############

        speaker_metrics = default_metrics(gpus)
        results = {'objective': {'raw': {}}}

        for key, value in files.items():
            results['objective']['raw'][key] = []

            utterance_embeds = []

            for file in value:

                # Get prosody metrics
                file_metrics = promonet.evaluate.Metrics(gpu)

                # Get target filepath
                target_prefix = \
                    original_objective_directory / file.parent.name / file.stem

                # Get predicted filepath
                predicted_prefix = \
                    objective_directory / file.parent.name / file.stem

                # Get waveform filepath
                waveform_prefix = \
                    subjective_directory / file.parent.name / file.stem

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

                # import pdb; pdb.set_trace()

                # Get predicted PPGs
                size = prosody_args[0].shape[-1]
                mode = promonet.PPG_INTERP_METHOD
                ppg_model = (
                    'ppg' if promonet.PPG_MODEL is None else
                    f'{promonet.PPG_MODEL}')
                predicted_ppgs = torch.nn.functional.interpolate(
                    load_ppg_file(f'{predicted_prefix}-{ppg_model}.pt', device)[None],
                    size=size,
                    mode=mode,
                    align_corners=None if mode == 'nearest' else False)[0]

                # Get target PPGs
                target_ppgs = torch.nn.functional.interpolate(
                    load_ppg_file(f'{target_prefix}-{ppg_model}.pt', device)[None],
                    size=size,
                    mode=mode,
                    align_corners=None if mode == 'nearest' else False)[0]

                ppg_args = (predicted_ppgs, target_ppgs)
                condition = '-'.join(target_prefix.stem.split('-')[1:3])

                #Get target text and audio file for WER
                #TODO do these files exist?
                gt_text = promonet.load.text(f'{target_prefix}-text.txt')
                audio_file = f'{waveform_prefix}.wav'

                wer_args = (gt_text, audio_file)

                speaker_embed = get_speaker_embed(dataset, file.parent.name, gpu)
                utterance_embed = get_utterance_embed(audio_file, gpu)
                utterance_embeds.append(utterance_embed)

                speaker_sim_args = (speaker_embed, utterance_embed)

                # Update metrics
                metrics[condition].update(prosody_args, ppg_args, wer_args, speaker_sim_args)

                speaker_metrics[condition].update(prosody_args, ppg_args, wer_args, speaker_sim_args)
                file_metrics.update(prosody_args, ppg_args, wer_args, speaker_sim_args)

                # Get results for this file
                results['objective']['raw'][key].append(
                    (file.stem, file_metrics()))

                # Reset prosody metrics
                file_metrics.reset()
            
            if key == 'original':
                speaker = value[0].parent.name
                gt_embed = get_speaker_embed(dataset, speaker, gpu, return_all=True)
                filename = f'{dataset}-{speaker}.pdf'
                promonet.plot.speaker_cluster.plot_speaker_clusters(np.array(gt_embed), np.array(utterance_embeds), [speaker for i in range(len(gt_embed))], [speaker for i in range(len(utterance_embeds))], file = plot_dir / filename)

        # Get results for this speaker
        results['objective']['average'] = {
            key: value() for key, value in speaker_metrics.items()}
        

    # Get the total number of samples we have generated
    files = subjective_directory.rglob('*.wav')
    results['num_samples'] = sum(
        [torchaudio.info(file).num_frames for file in files])
    results['num_frames'] = promonet.convert.samples_to_frames(
        results['num_samples'])

    # Print results and save to disk
    print(results)
    with open(objective_directory / index / 'results.json', 'w') as file:
        json.dump(results, file, indent=4, sort_keys=True)


###############################################################################
# Utilities
###############################################################################

def default_metrics(gpus):
    """Construct the default metrics dictionary for each condition"""
    # Bind shared parameter
    gpu = None if gpus is None else gpus[0]
    metric_fn = functools.partial(promonet.evaluate.Metrics, gpu)

    # Reconstruction metrics
    metrics = {'original-100': metric_fn()}

    if promonet.MODEL not in ['hifigan', 'vits']:

        # Prosody editing metrics
        for condition in ['scaled', 'shifted', 'stretched']:
            for ratio in RATIOS:
                metrics[f'{condition}-{int(ratio * 100):03d}'] = metric_fn()

    return metrics

def load_ppg_file(ppg_file, device):
    if promonet.PPG_MODEL is not None:
        ppg = torch.load(ppg_file).to(device)
        if ppgs.FRONTEND is not None and '-ppg' not in promonet.PPG_MODEL:
            if not hasattr(load_ppg_file, 'frontend'):
                load_ppg_file.frontend = ppgs.FRONTEND(device)
            ppg = load_ppg_file.frontend(ppg.unsqueeze(dim=0)).squeeze(dim=0)
        return ppg.to(torch.float32)
    else:
        return torch.load(ppg_file).to(device)

def get_resemblyzer(gpu):
    if not hasattr(get_resemblyzer, 'encoder') or gpu != get_resemblyzer.gpu:
        device = 'cpu' if gpu is None else f'cuda:{gpu}'
        encoder = resemblyzer.VoiceEncoder(device=device)
        get_resemblyzer.encoder = encoder
        get_resemblyzer.gpu = gpu
    return get_resemblyzer.encoder

def get_speaker_embed(dataset, speaker, gpu, return_all=False): #return_all true -> return list of all embeddings, not speaker embedding
    model = get_resemblyzer(gpu)
    if not hasattr(get_speaker_embed, 'dictionary'):
        get_speaker_embed.dictionary = {}
    if not hasattr(get_speaker_embed, 'all_dictionary'):
        get_speaker_embed.all_dictionary = {}
    speaker_code = f'{dataset}-{speaker}'
    if speaker_code not in get_speaker_embed.dictionary.keys(): #Don't already have this embedding
        all_gts = (promonet.CACHE_DIR / dataset / speaker).glob('*-100.wav')
        gt_wavs = [resemblyzer.preprocess_wav(file) for file in all_gts]
        get_speaker_embed.dictionary[speaker_code] = model.embed_speaker(gt_wavs)
        get_speaker_embed.all_dictionary[speaker_code] = [model.embed_utterance(wav) for wav in gt_wavs]
        
    return get_speaker_embed.dictionary[speaker_code] if not return_all else get_speaker_embed.all_dictionary[speaker_code]

def get_utterance_embed(utterance, gpu):
    model = get_resemblyzer(gpu)
    if type(utterance) is torch.Tensor:
        utterance_preprocess = resemblyzer.preprocess_wav(utterance.numpy(force=True))
    else: #Should be fpath
        utterance_preprocess = resemblyzer.preprocess_wav(utterance)
    utterance_embed = model.embed_utterance(utterance_preprocess)
    return utterance_embed