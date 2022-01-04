import pysodic

import promovits


###############################################################################
# Constants
###############################################################################


ALL_FEATURES = ['ppg', 'prosody', 'text', 'spectrogram']


###############################################################################
# Data preprocessing
###############################################################################


def datasets(datasets, features=ALL_FEATURES, gpu=None):
    """Preprocess a dataset"""
    for dataset in datasets:

        # Get cache directory
        directory = promovits.CACHE_DIR / dataset

        # Iterate over speakers
        for speaker_directory in directory.glob('*'):

            # Get text and audio files for this speaker
            text_files = sorted(list(speaker_directory.rglob('*.txt')))
            audio_files = sorted(list(speaker_directory.glob('*.wav')))

            # Preprocess files
            from_files_to_files(
                speaker_directory,
                audio_files,
                text_files,
                features,
                gpu)


def from_files_to_files(
    output_directory,
    audio_files,
    text_files=None,
    features=ALL_FEATURES,
    gpu=None):
    """Preprocess from files"""
    # Change directory
    with promovits.data.chdir(output_directory):

        # Preprocess phonemes from text
        # TEMPORARY - text preprocessing is causing deadlock
        # if 'phonemes' in features:
        #     phoneme_files = [
        #         f'{file.stem}-text.pt' for file in text_files]
        #     promovits.preprocess.text.from_files_to_files(
        #         text_files,
        #         phoneme_files)

        # Preprocess spectrograms
        if 'spectrogram' in features:
            spectrogram_files = [
                f'{file.stem}-spectrogram.pt' for file in audio_files]
            promovits.preprocess.spectrogram.from_files_to_files(
                audio_files,
                spectrogram_files)

        # Preprocess phonetic posteriorgrams
        if 'ppg' in features:
            ppg_files = [f'{file.stem}-ppg.pt' for file in audio_files]
            promovits.preprocess.ppg.from_files_to_files(
                audio_files,
                ppg_files,
                gpu)

        # Preprocess prosody features
        if 'prosody' in features:
            prefixes = [file.stem for file in audio_files]
            pysodic.from_files_to_files(
                audio_files,
                prefixes,
                text_files,
                promovits.HOPSIZE / promovits.SAMPLE_RATE,
                promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
                gpu)


def prosody(audio, sample_rate=promovits.SAMPLE_RATE, text=None, gpu=None):
    """Preprocess prosody from audio to retrieve features for editing"""
    hopsize = promovits.HOPSIZE / promovits.SAMPLE_RATE
    window_size = promovits.WINDOW_SIZE / promovits.SAMPLE_RATE

    # Get prosody features including alignment
    if text:
        output = pysodic.from_audio_and_text(
            audio,
            sample_rate,
            text,
            hopsize,
            window_size,
            gpu)

        # Pitch, loudness and alignment
        return output[0], output[2], output[5]

    # Get prosody features without alignment
    output = pysodic.from_audio(audio, sample_rate, hopsize, window_size, gpu)

    # Pitch and loudness
    return output[0], output[2]
