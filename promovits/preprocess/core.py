import pysodic

import promovits


###############################################################################
# Constants
###############################################################################


ALL_FEATURES = ['phonemes', 'pitch', 'ppg', 'spectrogram']


###############################################################################
# Data preprocessing
###############################################################################


def datasets(datasets, features=ALL_FEATURES, gpu=None):
    """Preprocess a dataset"""
    for dataset in datasets:

        # Get cache directory
        directory = promovits.CACHE_DIR / dataset

        # Get text and audio files
        text_files = sorted(list(directory.glob('*.txt')))
        audio_files = sorted(list(directory.glob('*.wav')))

        # Change directory
        with promovits.data.chdir(directory):

            # Preprocess phonemes from text
            # TEMPORARY - text preprocessing is causing deadlock
            # if 'phonemes' in features:
            #     phoneme_files = [
            #         f'{file.stem}-phonemes.pt' for file in text_files]
            #     promovits.preprocess.text.from_files_to_files(
            #         text_files,
            #         phoneme_files)

            # Preprocess spectrograms
            # if 'spectrogram' in features:
            #     spectrogram_files = [
            #         f'{file.stem}-spectrogram.pt' for file in audio_files]
            #     promovits.preprocess.spectrogram.from_files_to_files(
            #         audio_files,
            #         spectrogram_files)

            # Preprocess phonetic posteriorgrams
            if 'ppg' in features:
                ppg_files = [f'{file.stem}-ppg.pt' for file in audio_files]
                promovits.preprocess.ppg.from_files_to_files(
                    audio_files,
                    ppg_files,
                    gpu)

            # Preprocess pitch, periodicity, and loudness
            # if 'pitch' in features:
            #     prefixes = [file.stem for file in audio_files]
            #     pysodic.from_files_to_files(
            #         audio_files,
            #         prefixes,
            #         text_files,
            #         promovits.HOPSIZE / promovits.SAMPLE_RATE,
            #         promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
            #         gpu)
