import json

import torch
import torchutil

import promonet


###############################################################################
# Data augmentation
###############################################################################


@torchutil.notify('augment')
def datasets(datasets):
    """Perform data augmentation on cached datasets"""
    for dataset in datasets:

        # Remove cached metadata that may become stale
        for stats_file in (promonet.ASSETS_DIR / 'stats').glob('*.pt'):
            stats_file.unlink()

        # Get cache directory
        directory = promonet.CACHE_DIR / dataset

        # Get files
        audio_files = sorted(directory.rglob('*-100.wav'))

        # Augment
        from_files_to_files(audio_files, dataset)


def from_files_to_files(audio_files, name):
    """Perform data augmentation on audio files"""
    torch.manual_seed(promonet.RANDOM_SEED)

    # Get augmentation ratios
    ratios = sample(len(audio_files))

    # Get locations to save output
    output_files = [
        file.parent /
        f'{file.stem.split("-")[0]}-p{int(ratio * 100):03d}.wav'
        for file, ratio in zip(audio_files, ratios)]

    # Augment
    promonet.data.augment.pitch.from_files_to_files(
        audio_files,
        output_files,
        ratios)

    # Save augmentation ratios
    save(promonet.AUGMENT_DIR / f'{name}-pitch.json', audio_files, ratios)

    # Get augmentation ratios
    ratios = sample(len(audio_files))

    # Get locations to save output
    output_files = [
        file.parent /
        f'{file.stem.split("-")[0]}-l{int(ratio * 100):03d}.wav'
        for file, ratio in zip(audio_files, ratios)]

    # Augment
    # N.B. Ratios that cause clipping will be resampled
    ratios = promonet.data.augment.loudness.from_files_to_files(
        audio_files,
        output_files,
        ratios)

    # Save augmentation ratios
    save(
        promonet.AUGMENT_DIR / f'{name}-loudness.json',
        audio_files,
        ratios)


###############################################################################
# Data augmentation
###############################################################################


def sample(n):
    """Sample data augmentation ratios"""
    distribution = torch.distributions.uniform.Uniform(
        torch.log2(torch.tensor(promonet.AUGMENTATION_RATIO_MIN)),
        torch.log2(torch.tensor(promonet.AUGMENTATION_RATIO_MAX)))
    ratios = 2 ** distribution.sample([n])

    # Prevent duplicates
    ratios[(ratios * 100).to(torch.int) == 100] += 1

    return ratios


def save(json_file, audio_files, ratios):
    """Cache augmentation ratios"""
    ratio_dict = {}
    for audio_file, ratio in zip(audio_files, ratios):
        key = f'{audio_file.parent.name}/{audio_file.stem.split("-")[0]}'
        ratio_dict[key] = f'{int(ratio * 100):03d}'
    with open(json_file, 'w') as file:
        json.dump(ratio_dict, file, indent=4)
