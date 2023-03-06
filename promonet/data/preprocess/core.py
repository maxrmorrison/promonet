from pathlib import Path

import pysodic

import promonet
import ppgs

###############################################################################
# Constants
###############################################################################


ALL_FEATURES = ['ppg', 'prosody', 'phonemes', 'spectrogram'] + ppgs.REPRESENTATION_MAP.keys()

PPG_DIR = Path.home() / 'ppgs' / 'ppgs' / 'assets' #TODO fix this to not be... this

###############################################################################
# Data preprocessing
###############################################################################


def datasets(datasets, features=ALL_FEATURES, gpu=None):
    """Preprocess a dataset"""
    for dataset in datasets:

        # Get cache directory
        directory = promonet.CACHE_DIR / dataset

        # Iterate over speakers
        for speaker_directory in directory.glob('*'):

            # Get text and audio files for this speaker
            text_files = sorted(list(speaker_directory.rglob('*.txt')))
            audio_files = sorted(list(speaker_directory.rglob('*.wav')))
            audio_files = [
                file for file in audio_files if '-template' not in file.stem]

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
    with promonet.data.chdir(output_directory):

        for feature in features:
            if feature in ppgs.REPRESENTATION_MAP.keys():
                base_files = [f'{file.stem}-{feature}-base.pt' for file in audio_files]
                phoneme_files = [f'{file.stem}-{feature}-phoneme.pt' for file in audio_files]
                ppgs.from_files_to_files(audio_files, base_files, representation=feature, preprocess_only=True, gpu=gpu)
                ppgs.from_files_to_files(audio_files, phoneme_files, representation=feature, preprocess_only=False, gpu=gpu)


        # Preprocess phonemes from text
        if 'phonemes' in features:
            phoneme_files = [
                f'{file.stem}-text.pt' for file in text_files]
            promonet.data.preprocess.text.from_files_to_files(
                text_files,
                phoneme_files)

        # Preprocess spectrograms
        if 'spectrogram' in features:
            spectrogram_files = [
                f'{file.stem}-spectrogram.pt' for file in audio_files]
            promonet.data.preprocess.spectrogram.from_files_to_files(
                audio_files,
                spectrogram_files)

        # Preprocess phonetic posteriorgrams
        if 'ppg' in features:
            ppg_files = [f'{file.stem}-ppg.pt' for file in audio_files]
            promonet.data.preprocess.ppg.from_files_to_files(
                audio_files,
                ppg_files,
                gpu)

        # Preprocess prosody features
        if 'prosody' in features:
            prefixes = [Path(file.stem) for file in audio_files]
            pysodic.from_files_to_files(
                audio_files,
                prefixes,
                text_files,
                promonet.HOPSIZE / promonet.SAMPLE_RATE,
                promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
                gpu)

        # Template waveform
        if 'template' in features:
            prefixes = [Path(file.stem) for file in audio_files]
            promonet.data.preprocess.template.from_files_to_files(prefixes)


def prosody(audio, sample_rate=promonet.SAMPLE_RATE, text=None, gpu=None):
    """Preprocess prosody from audio to retrieve features for editing"""
    hopsize = promonet.HOPSIZE / promonet.SAMPLE_RATE
    window_size = promonet.WINDOW_SIZE / promonet.SAMPLE_RATE

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
