import argparse
import shutil
import ssl
import tarfile
import urllib
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
    # Download
    url = 'https://zenodo.org/record/4783456/files/daps-segmented.tar.gz?download=1'
    file = promovits.DATA_DIR / 'daps.tar.gz'
    download_file(url, file)

    with promovits.data.chdir(promovits.DATA_DIR):

        # Unzip
        with tarfile.open(file, 'r:gz') as tfile:
            tfile.extractall()

        # Rename directory
        if Path('daps').exists():
            shutil.rmtree('daps')
        shutil.move('daps-segmented', 'daps')

        # Get audio files
        audio_files = [path.resolve() for path in  Path('daps').rglob('*.wav')]
        text_files = [file.with_suffix('.txt') for file in audio_files]

    # Write audio to cache
    speaker_count = {}
    output_directory = promovits.CACHE_DIR / 'daps'
    output_directory.mkdir(exist_ok=True, parents=True)
    with promovits.data.chdir(output_directory):

        # Iterate over files
        iterator = tqdm.tqdm(
            zip(audio_files, text_files),
            desc='Formatting daps',
            dynamic_ncols=True,
            total=len(audio_files))
        for audio_file, text_file in iterator:

            # Get speaker ID
            speaker = Path(audio_file.stem.split('_')[0])
            if speaker not in speaker_count:

                # Each entry is (index, count)
                speaker_count[speaker] = [len(speaker_count), 0]

            # Update speaker and get current entry
            speaker_count[speaker][1] += 1
            index, count = speaker_count[speaker]

            # Convert to 22.05k
            audio = promovits.load.audio(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # # Save to disk
            output_audio_file = f'{index:04d}-{count:06d}.wav'
            torchaudio.save(
                output_directory / output_audio_file,
                audio,
                promovits.SAMPLE_RATE)
            shutil.copyfile(
                text_file,
                (output_directory / output_audio_file).with_suffix('.txt'))

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

    # Get source files
    audio_files = sorted(list(audio_directory.rglob('*.flac')))
    text_files = [vctk_audio_file_to_text_file(file) for file in audio_files]

    # If the text file doesn't exist, remove corresponding audio file
    text_files = [file for file in text_files if file.exists()]
    audio_files = [
        file for file in audio_files
        if vctk_audio_file_to_text_file(file).exists()]

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

            # Get speaker ID
            speaker = Path(audio_file.stem.split('_')[0])
            if speaker not in speaker_count:

                # Each entry is (index, count)
                speaker_count[speaker] = [len(speaker_count), 0]

            # Update speaker and get current entry
            speaker_count[speaker][1] += 1
            index, count = speaker_count[speaker]

            # Convert to 22.05k wav
            audio = promovits.load.audio(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # Save to disk
            output_audio_file = f'{index:04d}-{count:06d}.wav'
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


def vctk_audio_file_to_text_file(audio_file):
    """Convert audio file to corresponding text file"""
    text_directory = promovits.DATA_DIR / 'vctk' / 'txt'
    return (
        text_directory /
        audio_file.parent.name /
        f'{audio_file.stem[:-5]}.txt')


def vctk_text_file_to_audio_file(text_file):
    """Convert audio file to corresponding text file"""
    audio_directory = promovits.DATA_DIR / 'vctk' / 'wav48_silence_trimmed'
    return (
        audio_directory /
        text_file.parent.name /
        f'{text_file.stem}.flac')


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
