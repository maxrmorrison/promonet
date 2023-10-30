"""Download and format datasets

Files produced are saved in data/. The directory structure is as follows.

data
├── cache
|   └── <dataset>
|       ├── <speaker>
|       |   ├── <utterance>-<ratio>-loudness.pt
|       |   ├── <utterance>-<ratio>-periodicity.pt
|       |   ├── <utterance>-<ratio>-pitch.pt
|       |   ├── <utterance>-<ratio>-ppg.pt
|       |   ├── <utterance>-<ratio>-spectrogram.pt
|       |   ├── <utterance>-<ratio>.wav
|       |   ├── <utterance>.txt
|       |   └── <utterance>.wav
|       └── <partition>-lengths.json  # Cached utterance lengths
├── datasets
|   └── <dataset>
|       └── <original, uncompressed contents of the dataset>
└── sources
    └── <dataset>
        └── <tar or zip files>
"""
import json
import shutil
import zipfile
from pathlib import Path

import torch
import torchaudio
import torchutil

import promonet


###############################################################################
# Download datasets
###############################################################################


@torchutil.notify.on_return('download')
def datasets(datasets):
    """Download datasets"""
    # Download and format daps dataset
    if 'daps' in datasets:
        daps()

    # Download and format libritts dataset
    if 'libritts' in datasets:
        libritts()

    # Download and format vctk dataset
    if 'vctk' in datasets:
        vctk()


def daps():
    """Download daps dataset"""
    torchutil.download.targz(
        'https://zenodo.org/record/4783456/files/daps-segmented.tar.gz?download=1',
        promonet.DATA_DIR)

    # Delete previous directory
    shutil.rmtree(promonet.DATA_DIR / 'daps', ignore_errors=True)

    # Rename directory
    data_directory = promonet.DATA_DIR / 'daps'
    shutil.move(
        promonet.DATA_DIR / 'daps-segmented',
        data_directory)

    # Get audio files
    audio_files = sorted(
        [path.resolve() for path in data_directory.rglob('*.wav')])
    text_files = [file.with_suffix('.txt') for file in audio_files]

    # Write audio to cache
    speaker_count = {}
    cache_directory = promonet.CACHE_DIR / 'daps'
    cache_directory.mkdir(exist_ok=True, parents=True)
    with promonet.chdir(cache_directory):

        # Iterate over files
        for audio_file, text_file in promonet.iterator(
            zip(audio_files, text_files),
            'Formatting daps',
            total=len(audio_files)
        ):

            # Get speaker ID
            speaker = Path(audio_file.stem.split('_')[0])
            if speaker not in speaker_count:

                # Each entry is (index, count)
                speaker_count[speaker] = [len(speaker_count), 0]

            # Update speaker and get current entry
            speaker_count[speaker][1] += 1
            index, count = speaker_count[speaker]

            # Load audio
            audio, sample_rate = torchaudio.load(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # Save at original sampling rate
            speaker_directory = cache_directory / f'{index:04d}'
            speaker_directory.mkdir(exist_ok=True, parents=True)
            output_file = Path(f'{count:06d}.wav')
            torchaudio.save(
                speaker_directory / output_file,
                audio,
                sample_rate)
            shutil.copyfile(
                text_file,
                (speaker_directory / output_file).with_suffix('.txt'))

            # Save at system sample rate
            audio = promonet.resample(audio, sample_rate)
            torchaudio.save(
                speaker_directory / f'{output_file.stem}-100.wav',
                audio,
                promonet.SAMPLE_RATE)


def libritts():
    """Download libritts dataset"""
    # Create directory for downloads
    source_directory = promonet.SOURCE_DIR / 'libritts'
    source_directory.mkdir(exist_ok=True, parents=True)

    # Create directory for unpacking
    data_directory = promonet.DATA_DIR / 'libritts'
    data_directory.mkdir(exist_ok=True, parents=True)

    # Download and unpack
    for partition in [
        'train-clean-100',
        'train-clean-360',
        'dev-clean',
        'test-clean']:
        torchutil.download.targz(
            f'https://us.openslr.org/resources/60/{partition}.tar.gz',
            promonet.DATA_DIR)

    # Uncapitalize directory name
    shutil.rmtree(str(data_directory), ignore_errors=True)
    shutil.move(
        str(promonet.DATA_DIR / 'LibriTTS'),
        str(data_directory),
        copy_function=shutil.copytree)

    # File locations
    audio_files = sorted(data_directory.rglob('*.wav'))
    text_files = [
        file.with_suffix('.normalized.txt') for file in audio_files]

    # Write audio to cache
    speaker_count = {}
    cache_directory = promonet.CACHE_DIR / 'libritts'
    cache_directory.mkdir(exist_ok=True, parents=True)
    with promonet.chdir(cache_directory):

        # Iterate over files
        for audio_file, text_file in promonet.iterator(
            zip(audio_files, text_files),
            'Formatting libritts',
            total=len(audio_files)
        ):

            # Get file metadata
            speaker, *_ = [
                int(part) for part in audio_file.stem.split('_')]

            # Update speaker and get current entry
            if speaker not in speaker_count:

                # Each entry is (index, count)
                speaker_count[speaker] = [len(speaker_count), 0]

            speaker_count[speaker][1] += 1
            index, count = speaker_count[speaker]

            # Load audio
            audio, sample_rate = torchaudio.load(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # Save at system sample rate
            stem = f'{index:04d}/{count:06d}'
            output_file = Path(f'{stem}.wav')
            output_file.parent.mkdir(exist_ok=True, parents=True)
            audio = promonet.resample(audio, sample_rate)
            torchaudio.save(
                output_file.parent / f'{output_file.stem}-100.wav',
                audio,
                promonet.SAMPLE_RATE)
            shutil.copyfile(text_file, output_file.with_suffix('.txt'))

        # Save speaker map
        with open('speakers.json', 'w') as file:
            json.dump(speaker_count, file, indent=4, sort_keys=True)


def vctk():
    """Download vctk dataset"""
    directory = promonet.DATA_DIR / 'vctk'
    directory.mkdir(exist_ok=True, parents=True)
    torchutil.download.zip(
        'https://datashare.ed.ac.uk/download/DS_10283_3443.zip',
        directory)

    # Unzip
    for file in directory.glob('*.zip'):
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
    output_directory = promonet.CACHE_DIR / 'vctk'
    output_directory.mkdir(exist_ok=True, parents=True)
    with promonet.chdir(output_directory):

        # Iterate over files
        for audio_file, text_file in promonet.iterator(
            zip(audio_files, text_files),
            'Formatting vctk',
            total=len(audio_files)
        ):

            # Get speaker ID
            speaker = Path(audio_file.stem.split('_')[0])
            if speaker not in speaker_count:

                # Each entry is (index, count)
                speaker_count[speaker] = [len(speaker_count), 0]

            # Update speaker and get current entry
            speaker_count[speaker][1] += 1
            index, count = speaker_count[speaker]

            # Load audio
            audio, sample_rate = torchaudio.load(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # Save at original sampling rate
            speaker_directory = output_directory / f'{index:04d}'
            speaker_directory.mkdir(exist_ok=True, parents=True)
            output_file = Path(f'{count:06d}.wav')
            torchaudio.save(
                speaker_directory / output_file,
                audio,
                sample_rate)
            shutil.copyfile(
                text_file,
                (speaker_directory / output_file).with_suffix('.txt'))

            # Save at system sample rate
            audio = promonet.resample(audio, sample_rate)
            torchaudio.save(
                speaker_directory / f'{output_file.stem}-100.wav',
                audio,
                promonet.SAMPLE_RATE)


###############################################################################
# Utilities
###############################################################################


def vctk_audio_file_to_text_file(audio_file):
    """Convert audio file to corresponding text file"""
    text_directory = promonet.DATA_DIR / 'vctk' / 'txt'
    return (
        text_directory /
        audio_file.parent.name /
        f'{audio_file.stem[:-5]}.txt')


def vctk_text_file_to_audio_file(text_file):
    """Convert audio file to corresponding text file"""
    audio_directory = promonet.DATA_DIR / 'vctk' / 'wav48_silence_trimmed'
    return (
        audio_directory /
        text_file.parent.name /
        f'{text_file.stem}.flac')
