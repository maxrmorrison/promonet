import torchutil

import promonet


###############################################################################
# Preprocess datasets
###############################################################################


@torchutil.notify.on_return('preprocess')
def datasets(datasets, features=promonet.ALL_FEATURES, gpu=None):
    """Preprocess a dataset"""
    for dataset in datasets:

        # Get cache directory
        directory = promonet.CACHE_DIR / dataset

        # Get text and audio files for this speaker
        audio_files = sorted(list(directory.rglob('*.wav')))
        audio_files = [file for file in audio_files if '-' in file.stem]
        text_files = [
            file.parent / f'{file.stem[:-4]}.txt' for file in audio_files]

        # Preprocess input features
        if any(feature in features for feature in [
            'loudness',
            'periodicity',
            'pitch',
            'ppg'
        ]):
            promonet.preprocess.from_files_to_files(
                audio_files,
                features=features,
                gpu=gpu)

        # Preprocess spectrograms
        if 'spectrogram' in features:
            spectrogram_files = [
                file.parent / f'{file.stem}-spectrogram.pt'
                for file in audio_files]
            promonet.preprocess.spectrogram.from_files_to_files(
                audio_files,
                spectrogram_files)

        # Preprocess alignment
        if 'alignment' in features:
            prefixes = [file.name / file.stem for file in text_files]
            promonet.preprocess.forced_alignment(
                text_files,
                audio_files,
                prefixes)
