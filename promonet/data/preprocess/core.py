import functools
import multiprocessing as mp
import os
from pathlib import Path

import ppgs
import pyfoal
import pypar
import pysodic
import torchutil
import torchaudio

import promonet


###############################################################################
# Data preprocessing
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

        # Preprocess files
        from_files_to_files(audio_files, text_files, features, gpu)


def from_files_to_files(
    audio_files,
    text_files=None,
    features=promonet.ALL_FEATURES,
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
        if '-latents' in promonet.PPG_MODEL:
            latent_files = [
                file.parent / f'{file.stem}-{promonet.PPG_MODEL}.pt'
                for file in audio_files]
            ppgs.preprocess.from_files_to_files(
                audio_files,
                latent_files,
                features=[promonet.PPG_MODEL.split('-')[0]],
                gpu=gpu,
                num_workers=promonet.NUM_WORKERS)
        elif '-ppg' in promonet.PPG_MODEL:
            ppg_files = [
                file.parent / f'{file.stem}-{promonet.PPG_MODEL}.pt'
                for file in audio_files]
            ppgs.from_files_to_files(
                audio_files,
                ppg_files,
                num_workers=promonet.NUM_WORKERS,
                gpu=gpu)
        else:
            raise ValueError(f'Unknown PPG model "{promonet.PPG_MODEL}"')

    # Preprocess prosody features
    if 'prosody' in features:
        prefixes = [file.parent / file.stem for file in audio_files]
        hopsize = promonet.HOPSIZE / promonet.SAMPLE_RATE
        extraction_fn = functools.partial(
            pysodic.from_file_to_file,
            hopsize=hopsize,
            window_size=promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
            voicing_threshold=0.1625,
            gpu=gpu)
        for audio_file, output_prefix in promonet.iterator(
            zip(audio_files, prefixes),
            'pysodic',
            total=len(audio_files)
        ):
            extraction_fn(audio_file, output_prefix)

        # Extract forced alignments
        alignment_files = [
            Path(f'{prefix}.TextGrid') for prefix in prefixes]

        # Default to using all cpus
        num_workers = max(min(len(text_files) // 2, os.cpu_count() // 2), 1)

        # Launch multiprocessed P2FA alignment
        align_fn = functools.partial(pyfoal_catch_failed)
        iterator = zip(text_files, audio_files, alignment_files)
        with mp.get_context('spawn').Pool(num_workers) as pool:
            failed = pool.starmap(align_fn, iterator)

        # Handle alignment failures of data-augmented speech by stretching
        # non-augmented alignments
        for file in [i for i in failed if i is not None]:
            good_textgrid = file.parent / f'{file.stem[:-4]}-100.TextGrid'
            ratio = int(file.stem[-3:]) / 100.
            alignment = pypar.Alignment(good_textgrid)
            durations = [
                phoneme.duration() for phoneme in alignment.phonemes()]
            new_durations = [duration / ratio for duration in durations]
            alignment.update(durations=new_durations)
            alignment.save(file.parent / f'{file.stem}.TextGrid')

        # Get exact lengths derived from audio files to avoid length
        # mismatch due to floating-point vs integer hopsize
        lengths = [
            int(
                torchaudio.info(file).num_frames //
                (torchaudio.info(file).sample_rate * hopsize)
            )
            for file in audio_files]

        # Convert alignments to indices
        indices_files = [
            file.parent / f'{file.stem}-phonemes.pt'
            for file in alignment_files]
        pysodic.alignment_files_to_indices_files(
            alignment_files,
            indices_files,
            lengths,
            hopsize)


###############################################################################
# Utilities
###############################################################################


def pyfoal_catch_failed(text_file, audio_file, output_file):
    try:
        pyfoal.from_file_to_file(
            text_file,
            audio_file,
            output_file,
            aligner='p2fa')
    except IndexError:
        return audio_file
