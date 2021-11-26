import argparse
import urllib
import shutil
import ssl
import zipfile
from pathlib import Path

import torch
import torchaudio
import tqdm

import promovits


###############################################################################
# Download datasets
###############################################################################


def datasets_from_cloud(datasets):
    """Download datasets from cloud storage"""
    # Download and format vctk dataset
    if 'vctk' in datasets:
        vctk()


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
            speaker_dir = Path(audio_file.stem.split('_')[0])
            speaker_dir.mkdir(exist_ok=True, parents=True)

            # Convert to 22.05k wav
            audio = promovits.load.audio(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # Save to disk
            output_audio_file = f'{len(list(speaker_dir.glob("*"))):06d}.wav'
            torchaudio.save(
                speaker_dir / output_audio_file,
                audio,
                promovits.SAMPLE_RATE)
            shutil.copyfile(
                text_file,
                (speaker_dir / output_audio_file).with_suffix('.txt'))


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
    datasets_from_cloud(**vars(parse_args()))
