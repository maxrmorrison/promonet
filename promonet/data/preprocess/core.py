import ppgs
import pysodic
import penn

import promonet



###############################################################################
# Constants
###############################################################################


ALL_FEATURES = ['ppg', 'prosody', 'spectrogram']


###############################################################################
# Data preprocessing
###############################################################################

@promonet.notify.notify_on_finish('preprocess')
def datasets(datasets, features=ALL_FEATURES, gpu=None):
    """Preprocess a dataset"""
    for dataset in datasets:

        # Get cache directory
        directory = promonet.CACHE_DIR / dataset

        # Get text and audio files for this speaker
        audio_files = sorted(list(directory.rglob('*.wav')))
        audio_files = [file for file in audio_files if '-' in file.stem]
        text_files = [
            file.parent / f'{file.stem[:-4]}.txt' for file in audio_files]

        # Preprocess files
        from_files_to_files(audio_files, text_files, features, gpu)


def from_files_to_files(
    audio_files,
    text_files=None,
    features=ALL_FEATURES,
    gpu=None):
    """Preprocess from files"""
    # Preprocess spectrograms
    if 'spectrogram' in features:
        spectrogram_files = [
            file.parent / f'{file.stem}-spectrogram.pt'
            for file in audio_files]
        promonet.data.preprocess.spectrogram.from_files_to_files(
            audio_files,
            spectrogram_files)

    # Preprocess phonetic posteriorgrams
    if 'ppg' in features:
        if promonet.PPG_MODEL is None:
            ppg_files = [
                file.parent / f'{file.stem}-ppg.pt' for file in audio_files]
            promonet.data.preprocess.ppg.from_files_to_files(
                audio_files,
                ppg_files,
                gpu)
        else:
            if '-latents' in promonet.PPG_MODEL:
                latent_files = [
                file.parent / f'{file.stem}-{promonet.PPG_MODEL}.pt' for file in audio_files]
                ppgs.preprocess.from_files_to_files(
                    audio_files,
                    latent_files,
                    features=[promonet.PPG_MODEL.split('-')[0]],
                    gpu=gpu,
                    num_workers=promonet.NUM_WORKERS
                )
            elif '-ppg' in promonet.PPG_MODEL:
                ppg_files = [
                    file.parent / f'{file.stem}-{promonet.PPG_MODEL}.pt' for file in audio_files]
                ppgs.from_files_to_files(
                    audio_files,
                    ppg_files,
                    representation=ppgs.REPRESENTATION,
                    gpu=gpu,
                    num_workers=promonet.NUM_WORKERS
                )
            else:
                raise ValueError(f'unknown PPG_MODEL: {promonet.PPG_MODEL}')

    # Preprocess prosody features
    if 'prosody' in features:
        print(f"starting prosody preprocessing {len(audio_files)} audio files")
        prefixes = [file.parent / file.stem for file in audio_files]
        pysodic.from_files_to_files(
            audio_files,
            prefixes,
            # text_files,
            hopsize=promonet.HOPSIZE / promonet.SAMPLE_RATE,
            window_size=promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
            voicing_threshold=0.1625,
            gpu=gpu)
    if 'pitch' in features:
        print(f"starting pitch preprocessing {len(audio_files)} audio files")
        prefixes = [file.parent / file.stem for file in audio_files]
        penn.from_files_to_files(
            audio_files,
            prefixes,
            hopsize=promonet.HOPSIZE / promonet.SAMPLE_RATE,
            fmin=pysodic.FMIN,
            fmax=pysodic.FMAX,
            batch_size=1024,
            interp_unvoiced_at=0.1625,
            gpu=gpu
        )
