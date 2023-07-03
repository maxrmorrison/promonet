import shutil
import ssl
import tarfile
import urllib
import zipfile
from pathlib import Path

import torch
import torchaudio
import tqdm

import promonet


###############################################################################
# Download datasets
###############################################################################

@promonet.notify.notify_on_finish('download')
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
    # Download
    url = 'https://zenodo.org/record/4783456/files/daps-segmented.tar.gz?download=1'
    file = promonet.SOURCES_DIR / 'daps.tar.gz'
    download_file(url, file)

    with promonet.chdir(promonet.DATA_DIR):

        # Unzip
        with tarfile.open(file, 'r:gz') as tfile:
            tfile.extractall()

        # Rename directory
        if Path('daps').exists():
            shutil.rmtree('daps')
        shutil.move('daps-segmented', 'daps')

        # Get audio files
        audio_files = sorted(
            [path.resolve() for path in  Path('daps').rglob('*.wav')])
        text_files = [file.with_suffix('.txt') for file in audio_files]

    # Write audio to cache
    speaker_count = {}
    output_directory = promonet.CACHE_DIR / 'daps'
    output_directory.mkdir(exist_ok=True, parents=True)
    with promonet.chdir(output_directory):

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

        # Download
        url = f'https://us.openslr.org/resources/60/{partition}.tar.gz'
        file = source_directory / f'libritts-{partition}.tar.gz'
        download_file(url, file)

        # Unpack
        with tarfile.open(file, 'r:gz') as tfile:
            tfile.extractall(promonet.DATA_DIR)

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

    # Track previous and next utterances
    context = {}
    prev_parts = None

    # Write audio to cache
    speaker_count = {}
    cache_directory = promonet.CACHE_DIR / 'libritts'
    cache_directory.mkdir(exist_ok=True, parents=True)
    with promonet.chdir(cache_directory):

        # Iterate over files
        iterator = tqdm.tqdm(
            zip(audio_files, text_files),
            desc='Formatting libritts',
            dynamic_ncols=True,
            total=len(audio_files))
        for audio_file, text_file in iterator:

            # Get file metadata
            speaker, book, chapter, utterance = [
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
                output_file.parent / f'{output_file.stem}.wav',
                audio,
                promonet.SAMPLE_RATE)
            shutil.copyfile(text_file, output_file.with_suffix('.txt'))

            # Update context
            if (
                prev_parts is not None and
                prev_parts == (speaker, book, chapter, utterance - 1)
            ):
                prev_stem = f'{index:04d}/{count - 1:06d}'
                context[stem] = { 'prev': prev_stem, 'next': None }
                context[prev_stem]['next'] = stem
            else:
                context[stem] = { 'prev': None, 'next': None }
            prev_parts = (speaker, book, chapter, utterance)


def vctk():
    """Download vctk dataset"""
    # Download
    url = 'https://datashare.ed.ac.uk/download/DS_10283_3443.zip'
    file = promonet.SOURCES_DIR / 'vctk.zip'
    download_file(url, file)

    # Unzip
    directory = promonet.DATA_DIR / 'vctk'
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
    output_directory = promonet.CACHE_DIR / 'vctk'
    output_directory.mkdir(exist_ok=True, parents=True)
    with promonet.chdir(output_directory):

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


def download_file(url, file):
    """Download file from url"""
    with urllib.request.urlopen(url, context=ssl.SSLContext()) as response, \
         open(file, 'wb') as output:
        shutil.copyfileobj(response, output)
