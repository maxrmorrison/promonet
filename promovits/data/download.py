import argparse
import urllib
import shutil
import ssl
import tarfile
import zipfile
from pathlib import Path

import torch
import torchaudio
import tqdm

import promovits


###############################################################################
# Download datasets
###############################################################################


def datasets(datasets):
    """Download datasets"""
    # Download and format daps dataset
    if 'daps' in datasets:
        daps()

    # Download and format vctk dataset
    if 'vctk' in datasets:
        vctk()


def daps():
    """Download daps dataset"""
    # TODO - text files
    # TODO - organize by speaker

    # Download
    url = 'https://zenodo.org/record/4783456/files/daps-segmented.tar.gz?download=1'
    file = promovits.DATA_DIR / 'daps.tar.gz'
    download_file(url, file)

    # Unzip
    input_directory = promovits.DATA_DIR / 'daps'
    input_directory.mkdir(exist_ok=True, parents=True)
    with tarfile.open(file, 'r:gz') as tfile:
        tfile.extractall(input_directory)

    # Get audio files
    audio_files = list(input_directory.rglob('*.wav'))

    # Write audio to cache
    output_directory = promovits.CACHE_DIR / 'daps'
    output_directory.mkdir(exist_ok=True, parents=True)
    with promovits.data.chdir(output_directory):

        # Iterate over files
        iterator = tqdm.tqdm(
            enumerate(audio_files),
            desc='Formatting daps',
            dynamic_ncols=True,
            total=len(audio_files))
        for i, audio_file in iterator:

            # Convert to 22.05k
            audio = promovits.load.audio(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # Save to disk
            torchaudio.save(f'{i:06d}.wav', audio, promovits.SAMPLE_RATE)


def vctk():
    """Download vctk dataset"""
    # Download
    url = 'https://datashare.ed.ac.uk/download/DS_10283_3443.zip'
    file = promovits.DATA_DIR / 'vctk.zip'
    download_file(url, file)

    # Unzip
    directory = promovits.DATA_DIR / 'vctk'
    with zipfile.ZipFile(file, 'r') as zfile:
        zfile.extractall(directory)
    file = next((directory).glob('*.zip'))
    with zipfile.ZipFile(file) as zfile:
        zfile.extractall(directory)

    # File locations
    audio_directory = directory / 'wav48_silence_trimmed'
    text_directory = directory / 'txt'

    # Get source files
    audio_files = sorted(list(audio_directory.rglob('*.flac')))
    text_files = sorted(list(text_directory.rglob('*.txt')))

    # Write audio to cache
    speaker_count = {}
    output_directory = promovits.CACHE_DIR / 'vctk'
    output_directory.mkdir(exist_ok=True, parents=True)
    with promovits.data.chdir(output_directory):

        # Iterate over files
        iterator = tqdm.tqdm(
            zip(audio_files, text_files),
            desc='Formatting vctk',
            dynamic_ncols=True,
            total=len(audio_files))
        for audio_file, text_file in iterator:

            # Organize by speaker
            speaker = Path(audio_file.stem.split('_')[0])
            if speaker not in speaker_count:
                speaker_count[speaker] = 0
            index = speaker_count[speaker]
            speaker_count[speaker] += 1

            # Convert to 22.05k wav
            audio = promovits.load.audio(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # Save to disk
            output_audio_file = f'{speaker:04d}-{index:06d}.wav'
            torchaudio.save(
                output_directory / output_audio_file,
                audio,
                promovits.SAMPLE_RATE)
            shutil.copyfile(
                text_file,
                (output_directory / output_audio_file).with_suffix('.txt'))


###############################################################################
# Utilities
###############################################################################


def download_file(url, file):
    """Download file from url"""
    with urllib.request.urlopen(url, context=ssl.SSLContext()) as response, \
         open(file, 'wb') as output:
        shutil.copyfileobj(response, output)


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='The datasets to download')
    return parser.parse_args()


if __name__ == '__main__':
    datasets(**vars(parse_args()))
