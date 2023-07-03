import json
import multiprocessing as mp
import os

import resampy
import soundfile
import torch

import promonet


###############################################################################
# Data augmentation
###############################################################################


def datasets(datasets):
    """Perform data augmentation on cached datasets"""
    for dataset in datasets:

        # Get cache directory
        directory = promonet.CACHE_DIR / dataset

        # Get files
        audio_files = sorted(directory.rglob('*-100.wav'))

        # Augment and get augmentation ratios
        ratios = from_files_to_files(audio_files)

        # Save augmentation ratios
        with open(promonet.AUGMENT_DIR / f'{dataset}.json', 'w') as file:
            json.dump(ratios, file, indent=4)


def from_files_to_files(audio_files):
    """Perform data augmentation on audio files"""
    # Sample ratios
    torch.manual_seed(promonet.RANDOM_SEED)
    distribution = torch.distributions.uniform.Uniform(
        torch.log2(torch.tensor(promonet.AUGMENTATION_RATIO_MIN)),
        torch.log2(torch.tensor(promonet.AUGMENTATION_RATIO_MAX)))
    ratios = 2 ** distribution.sample([len(audio_files)])

    # Prevent duplicates
    ratios[(ratios * 100).to(torch.int) == 100] += 1

    # Perform multiprocessed augmentation
    iterator = list(zip(audio_files, ratios))
    with mp.get_context('spawn').Pool(os.cpu_count() // 2) as pool:
        pool.starmap(from_file_to_file, iterator)

    # Save augmentation info
    ratio_dict = {}
    for audio_file, ratio in iterator:
        key = \
            f'{audio_file.parent.name}/{audio_file.stem.split("-")[0]}'
        ratio_dict[key] = f'{int(ratio * 100):03d}'

    return ratio_dict


def from_file_to_file(audio_file, ratio):
    """Perform data augmentation on a file and save"""
    # Load audio
    audio, sample_rate = soundfile.read(str(audio_file))

    # Scale audio
    scaled = resampy.resample(
        audio,
        int(ratio * sample_rate),
        sample_rate)

    # Resample to promonet sample rate
    scaled = resampy.resample(scaled, sample_rate, promonet.SAMPLE_RATE)

    # Save to disk
    file = (
        audio_file.parent /
        f'{audio_file.stem.split("-")[0]}-{int(ratio * 100):03d}.wav')
    soundfile.write(str(file), scaled, promonet.SAMPLE_RATE)
