import json

import numpy as np
import torch
import torchaudio

import promonet
import ppgs


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, partition):
        super().__init__()
        self.metadata = Metadata(dataset, partition)
        self.cache = self.metadata.cache
        self.stems = self.metadata.stems
        self.lengths = self.metadata.lengths
        self.partition = partition

    def __getitem__(self, index):
        stem = self.stems[index]
        text = promonet.load.text(self.cache / f'{stem[:-4]}.txt')
        audio = promonet.load.audio(self.cache / f'{stem}.wav')
        pitch = torch.load(self.cache / f'{stem}-pitch.pt')
        periodicity = torch.load(self.cache / f'{stem}-periodicity.pt')
        loudness = torch.load(self.cache / f'{stem}-loudness.pt')
        spectrogram = torch.load(self.cache / f'{stem}-spectrogram.pt')

        # Get speaker index. Non-integer speaker names are assumed to be
        # for speaker adaptation and therefore default to index zero.
        if 'adapt' not in self.partition:
            speaker = int(stem.split('/')[0])
        else:
            speaker = 0

        # Load ppgs
        phonemes = promonet.load.ppg(
            self.cache / f'{stem}{ppgs.representation_file_extension()}',
            resample_length=spectrogram.shape[-1])

        # Data augmentation ratios
        augmentation = stem[-4:]
        if augmentation.startswith('-'):
            pitch_ratio, loudness_ratio = 1., 1.
        elif augmentation.startswith('p'):
            pitch_ratio = int(stem[-3:]) / 100.
            loudness_ratio = 1.
        elif augmentation.startswith('l'):
            pitch_ratio = 1.
            loudness_ratio = int(stem[-3:]) / 100.
        else:
            raise ValueError(
                f'Unrecognized augmentation string {augmentation}')

        return (
            text,
            phonemes,
            pitch,
            periodicity,
            loudness,
            spectrogram,
            audio,
            torch.tensor(speaker, dtype=torch.long),
            pitch_ratio,
            loudness_ratio,
            stem)

    def __len__(self):
        return len(self.stems)

    def buckets(self):
        """Partition indices into buckets based on length for sampling"""
        # Get the size of a bucket
        size = len(self) // promonet.BUCKETS

        # Get indices in order of length
        indices = np.argsort(self.lengths)
        lengths = np.sort(self.lengths)

        # Split into buckets based on length
        buckets = [
            np.stack((indices[i:i + size], lengths[i:i + size])).T
            for i in range(0, len(self), size)]

        # Concatenate partial bucket
        if len(buckets) == promonet.BUCKETS + 1:
            residual = buckets.pop()
            buckets[-1] = np.concatenate((buckets[-1], residual), axis=0)

        return buckets


###############################################################################
# Metadata
###############################################################################


class Metadata:

    def __init__(self, name, partition=None, overwrite_cache=False):
        """Create a metadata object for the given dataset or sources"""
        lengths = {}

        # Create dataset from string identifier
        self.name = name
        self.cache = promonet.CACHE_DIR / self.name

        # Get stems corresponding to partition
        partition_dict = promonet.load.partition(self.name)
        if partition is not None:
            stems = partition_dict[partition]
        else:
            stems = sum(partition_dict.values(), start=[])
        self.stems = [f'{stem}-100' for stem in stems]

        # For training, maybe add augmented data
        # This also applies to adaptation partitions: train-adapt-xx
        if 'train' in partition:
            if promonet.AUGMENT_PITCH:
                with open(
                    promonet.AUGMENT_DIR / f'{self.name}-pitch.json'
                ) as file:
                    ratios = json.load(file)
                self.stems.extend([f'{stem}-p{ratios[stem]}' for stem in stems])
            if promonet.AUGMENT_LOUDNESS:
                with open(
                    promonet.AUGMENT_DIR / f'{self.name}-loudness.json'
                ) as file:
                    ratios = json.load(file)
                self.stems.extend([f'{stem}-l{ratios[stem]}' for stem in stems])

        # Get audio filenames
        self.audio_files = [
            self.cache / (stem + '.wav') for stem in self.stems]

        # Get filename of cached lengths
        if partition is not None:
            lengths_file = self.cache / f'{partition}-lengths.json'
        else:
            lengths_file = self.cache / 'lengths.json'

        # Maybe remove existing cache data
        if overwrite_cache:
            lengths_file.unlink(missing_ok=True)

        # Load from cache
        if lengths_file.exists():
            with open(lengths_file, 'r') as file:
                lengths = json.load(file)

        self.lengths = []
        if not lengths:
            lengths = {}

        # Compute length in frames
        for stem, audio_file in zip(self.stems, self.audio_files):
            try:
                self.lengths.append(lengths[stem])
            except KeyError:
                info = torchaudio.info(audio_file)
                lengths[stem] = (
                    int(
                        info.num_frames *
                        (promonet.SAMPLE_RATE / info.sample_rate)
                    ) // promonet.HOPSIZE)
                self.lengths.append(lengths[stem])

        # Maybe cache lengths
        if self.cache is not None:
            with open(lengths_file, 'w+') as file:
                json.dump(lengths, file)

    def __len__(self):
        return len(self.stems)
