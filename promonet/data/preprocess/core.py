import torchutil

import promonet


###############################################################################
# Preprocess datasets
###############################################################################


@torchutil.notify('preprocess')
def datasets(datasets, features=promonet.ALL_FEATURES, gpu=None):
    """Preprocess a dataset"""
    for dataset in datasets:

        # Get cache directory
        directory = promonet.CACHE_DIR / dataset

        # Get text and audio files for this speaker
        audio_files = sorted(list(directory.rglob('*.wav')))
        audio_files = [file for file in audio_files if '-' in file.stem]

        # Preprocess input features
        if any(feature in features for feature in [
            'loudness',
            'pitch',
            'periodicity',
            'ppg',
            'text',
            'harmonics',
            'speaker'
        ]):
            promonet.preprocess.from_files_to_files(
                audio_files,
                gpu=gpu,
                features=[f for f in features if f != 'spectrogram'],
                loudness_bands=None)

        # Preprocess spectrograms
        if 'spectrogram' in features:
            spectrogram_files = [
                file.parent / f'{file.stem}-spectrogram.pt'
                for file in audio_files]
            promonet.preprocess.spectrogram.from_files_to_files(
                audio_files,
                spectrogram_files)
