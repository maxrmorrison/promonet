import functools
import json
import multiprocessing as mp

import resampy
import soundfile
import torch
import promovits
import tqdm


###############################################################################
# Constants
###############################################################################


MAX_RATIO = 2.
MIN_RATIO = .5


###############################################################################
# Data augmentation
###############################################################################


def datasets(datasets):
    """Perform data augmentation on cached datasets"""
    for dataset in datasets:

        # Get cache directory
        directory = promovits.CACHE_DIR / dataset

        # Seed random ratio sampling
        torch.manual_seed(promovits.RANDOM_SEED)

        # Save all augmentation ratios
        all_ratios = {}

        # Iterate over speakers
        directories = sorted(directory.glob('*'))
        speaker_iterator = tqdm.tqdm(
            directories,
            desc=f'Augmenting {dataset}',
            dynamic_ncols=True,
            total=len(directories))
        for speaker_directory in speaker_iterator:

            # Get text and audio files for this speaker
            audio_files = sorted(list(speaker_directory.rglob('*.wav')))
            audio_files = [
                file for file in audio_files if file.stem.endswith('-100')]
            text_files = sorted(list(speaker_directory.rglob('*.txt')))
            text_files = [
                file for file in text_files if file.stem.endswith('-100')]

            # Sample ratios
            distribution = torch.distributions.uniform.Uniform(
                torch.log2(torch.tensor(MIN_RATIO)),
                torch.log2(torch.tensor(MAX_RATIO)))
            ratios = 2 ** distribution.sample([len(audio_files)])

            # Prevent duplicates
            ratios[(ratios * 100).to(torch.int) == 100] += 1

            # Perform multiprocessed augmentation
            augment_fn = functools.partial(
                from_file_to_file,
                speaker_directory)
            augment_iterator = list(zip(audio_files, ratios))
            with mp.get_context('spawn').Pool() as pool:
                pool.starmap(augment_fn, augment_iterator)

            # Save augmentation info
            for audio_file, ratio in augment_iterator:
                key = \
                    f'{speaker_directory.stem}/{audio_file.stem.split("-")[0]}'
                all_ratios[key] = f'{int(ratio * 100):03d}'

        # Save all augmentation info to disk
        with open(promovits.AUGMENT_DIR / f'{dataset}.json', 'w') as file:
            json.dump(all_ratios, file, indent=4)


def from_file_to_file(directory, audio_file, ratio):
    """Perform data augmentation on a file and save"""
    # Load audio
    audio, sample_rate = soundfile.read(audio_file)

    # Scale audio
    scaled = resampy.resample(
        audio,
        int(ratio * sample_rate),
        sample_rate)

    # Resample to promovits sample rate
    scaled = resampy.resample(scaled, sample_rate, promovits.SAMPLE_RATE)

    # Save to disk
    file = (
        directory /
        f'{audio_file.stem.split("-")[0]}-{int(ratio * 100):03d}.wav')
    soundfile.write(file, scaled, promovits.SAMPLE_RATE)

