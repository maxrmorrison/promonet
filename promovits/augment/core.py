import functools
import json
import multiprocessing as mp
import torch
import torchaudio
import promovits

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
        for speaker_directory in directory.glob('*'):

            # Get text and audio files for this speaker
            audio_files = sorted(list(speaker_directory.rglob('*.wav')))

            # Sample ratios
            distribution = torch.distributions.uniform.Uniform(
                torch.log2(MIN_RATIO),
                torch.log2(MAX_RATIO))
            ratios = 2 ** distribution.sample(len(audio_files))

            # Perform multiprocessed augmentation
            augment_fn = functools.partial(
                from_file_to_file,
                speaker_directory)
            iterator = zip(audio_files, ratios)
            # TEMPORARY - remove MP for debugging
            # with mp.get_context('spawn').Pool() as pool:
            #     pool.starmap(augment_fn, iterator)
            for item in iterator:
                augment_fn(*item)

            # Save augmentation info
            for audio_file, ratio in iterator:
                key = \
                    f'{speaker_directory.stem}/{audio_file.stem.split("-")[0]}'
                all_ratios[key] = f'{int(ratio * 100):03d}'

        # Save all augmentation info to disk
        with open(promovits.AUGMENT_DIR / f'{dataset}.json', 'w') as file:
            json.dump(all_ratios, file, indent=4)


def from_file_to_file(directory, audio_file, ratio):
    """Perform data augmentation on a file and save"""
    # Load audio
    audio = promovits.load.audio(audio_file)

    # Scale audio
    scaled = promovits.resample(
        audio,
        ratio * promovits.SAMPLE_RATE,
        promovits.SAMPLE_RATE)

    # Resample to lpcnet sample rate
    scaled = promovits.resample(scaled, promovits.SAMPLE_RATE)

    # Save to disk
    file = f'{audio_file.stem.split("-")}-{int(ratio * 100):03d}.wav'
    torchaudio.save(directory / file, scaled, promovits.SAMPLE_RATE)

