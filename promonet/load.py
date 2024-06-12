import json

import ppgs
import torch
import torchaudio
import torchutil

import promonet


###############################################################################
# Loading utilities
###############################################################################


def audio(file):
    """Load audio from disk"""
    # Load
    audio, sample_rate = torchaudio.load(file)

    # Resample
    audio = torchaudio.functional.resample(
        audio,
        sample_rate,
        promonet.SAMPLE_RATE)

    # Ensure mono
    return audio.mean(dim=0, keepdims=True)


def features(prefix):
    """Load input features from file prefix"""
    if promonet.VITERBI_DECODE_PITCH:
        pitch_prefix = f'{prefix}-viterbi'
    else:
        pitch_prefix = prefix
    return (
        torch.load(f'{prefix}-loudness.pt'),
        torch.load(f'{pitch_prefix}-pitch.pt'),
        torch.load(f'{pitch_prefix}-periodicity.pt'),
        torch.load(f'{prefix}-ppg.pt'))


def partition(dataset, adapt=promonet.ADAPTATION):
    """Load partitions for dataset"""
    partition_dir = (
        promonet.ASSETS_DIR /
        'partitions' /
        ('adaptation' if adapt else 'multispeaker'))
    with open(partition_dir / f'{dataset}.json') as file:
        return json.load(file)


def pitch_distribution(dataset=promonet.TRAINING_DATASET, partition='train'):
    """Load pitch distribution"""
    if not hasattr(pitch_distribution, 'distribution'):

        # Location on disk
        key = ''
        if promonet.AUGMENT_LOUDNESS:
            key += '-loudness'
        if promonet.AUGMENT_PITCH:
            key += '-pitch'
        if promonet.VITERBI_DECODE_PITCH:
            key += '-viterbi'
        file = (
            promonet.ASSETS_DIR /
            'stats' /
            f'{dataset}-{promonet.PITCH_BINS}{key}.pt')

        if file.exists():

            # Load and cache distribution
            pitch_distribution.distribution = torch.load(file)

        else:

            # Get all voiced pitch frames
            allpitch = []
            dataset = promonet.data.Dataset(dataset, partition)
            viterbi = '-viterbi' if promonet.VITERBI_DECODE_PITCH else ''
            for stem in torchutil.iterator(
                dataset.stems,
                'promonet.load.pitch_distribution'
            ):
                pitch = torch.load(
                    dataset.cache / f'{stem}{viterbi}-pitch.pt')
                periodicity = torch.load(
                    dataset.cache / f'{stem}{viterbi}-periodicity.pt')
                allpitch.append(
                    pitch[
                        torch.logical_and(
                            ~torch.isnan(pitch),
                            periodicity > promonet.VOICING_THRESHOLD)])

            # Sort
            pitch, _ = torch.sort(torch.cat(allpitch))

            # Bucket
            indices = torch.linspace(
                len(pitch) / promonet.PITCH_BINS,
                len(pitch) - 1,
                promonet.PITCH_BINS,
                dtype=torch.float64
            ).to(torch.long)
            pitch_distribution.distribution = pitch[indices]

            # Save
            torch.save(pitch_distribution.distribution, file)

    return pitch_distribution.distribution


def per_speaker_averages(dataset=promonet.TRAINING_DATASET, partition='train'):
    """Load the average pitch in voiced regions for each speaker"""
    if not hasattr(per_speaker_averages, 'averages'):

        # Location on disk
        key = ''
        if promonet.VITERBI_DECODE_PITCH:
            key += '-viterbi'
        file = (
            promonet.ASSETS_DIR /
            'stats' /
            f'{dataset}-{partition}-speaker-averages{key}.json')

        try:

            # Load and cache averages
            with open(file) as json_file:
                per_speaker_averages.averages = json.load(json_file)

        except FileNotFoundError:

            # Get all voiced pitch frames
            allpitch = {}
            dataset = promonet.data.Dataset(dataset, partition)
            viterbi = '-viterbi' if promonet.VITERBI_DECODE_PITCH else ''
            for stem in torchutil.iterator(
                dataset.stems,
                'promonet.load.pitch_distribution'
            ):
                pitch = torch.load(
                    dataset.cache / f'{stem}{viterbi}-pitch.pt')
                periodicity = torch.load(
                    dataset.cache / f'{stem}{viterbi}-periodicity.pt')
                speaker = stem.split('/')[0]
                if speaker not in allpitch:
                    allpitch[speaker] = []
                allpitch[speaker].append(
                    pitch[
                        torch.logical_and(
                            ~torch.isnan(pitch),
                            periodicity > promonet.VOICING_THRESHOLD)])

            # Cache
            per_speaker_averages.averages = {
                speaker: 2 ** torch.log2(torch.cat(values)).mean().item()
                for speaker, values in allpitch.items()}

            # Save
            with open(file, 'w') as json_file:
                json.dump(
                    per_speaker_averages.averages,
                    json_file,
                    indent=4,
                    sort_keys=True)

    return per_speaker_averages.averages


def ppg(file, resample_length=None):
    """Load a PPG file and maybe resample"""
    # Load
    result = torch.load(file)

    # Maybe resample
    if resample_length is not None and result.shape[-1] != resample_length:
        result = promonet.edit.grid.sample(
            result,
            promonet.edit.grid.of_length(result, resample_length),
            promonet.PPG_INTERP_METHOD)

        # Preserve distribution
        if ppgs.REPRESENTATION_KIND == 'ppgs':
            return torch.softmax(torch.log(result + 1e-8), -2)

    return result


def text(file):
    """Load text file"""
    with open(file, encoding='utf-8') as file:
        return file.read()
