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
        self.dataset = dataset

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

        # Load supervised or unsupervised phoneme features
        if promonet.MODEL == 'vits':
            phonemes = promonet.load.phonemes(
                self.cache / f'{stem}-phonemes.pt',
                interleave=True)
        else:

            # Load ppgs
            phonemes = torch.load(self.cache / f'{stem}{ppgs.representation_file_extension()}')

            # Maybe resample length
            if phonemes.shape[1] != spectrogram.shape[-1]:
                grid = promonet.edit.grid.of_length(
                    phonemes,
                    spectrogram.shape[-1])
                phonemes = promonet.edit.grid.sample(
                    phonemes,
                    grid,
                    promonet.PPG_INTERP_METHOD)

        return (
            text,
            phonemes,
            pitch,
            periodicity,
            loudness,
            spectrogram,
            audio,
            torch.tensor(speaker, dtype=torch.long),
            int(stem[-3:]) / 100.,
            stem)

    def __len__(self):
        return len(self.stems)

    def buckets(self):
        """Partition indices into buckets based on length for sampling"""
        # Get the size of a bucket
        size = len(self) // promonet.BUCKETS

        # Get indices in order of length
        indices = np.argsort(self.lengths)

        # Split into buckets based on length
        buckets = [indices[i:i + size] for i in range(0, len(self), size)]

        # Add max length of each bucket
        return [(self.lengths[bucket[-1]], bucket) for bucket in buckets]


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
        if 'train' in partition and promonet.AUGMENT_PITCH:
            with open(promonet.AUGMENT_DIR / f'{self.name}.json') as file:
                ratios = json.load(file)
            self.stems.extend([f'{stem}-{ratios[stem]}' for stem in stems])

        # Maybe limit the maximum length during training to improve
        # GPU utilization
        if (
            ('train' in partition or 'valid' in partition) and
            promonet.MAX_TEXT_LENGTH is not None
        ):
            self.stems = [
                stem for stem in self.stems if (
                    # len(
                    #     promonet.load.phonemes(
                    #         self.cache / f'{stem}-phonemes.pt')
                    # ) < promonet.MAX_TEXT_LENGTH and
                    promonet.convert.samples_to_frames(
                        torchaudio.info(
                            self.cache / f'{stem}.wav').num_frames
                    ) < promonet.MAX_FRAME_LENGTH
                )
            ]

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

        if not lengths:

            # Compute length in frames
            for stem, audio_file in zip(self.stems, self.audio_files):
                lengths[stem] = \
                    torchaudio.info(audio_file).num_frames // ppgs.HOPSIZE

            # Maybe cache lengths
            if self.cache is not None:
                with open(lengths_file, 'w+') as file:
                    json.dump(lengths, file)

        # Match ordering
        self.lengths = [lengths[stem] for stem in self.stems]

    def __len__(self):
        return len(self.stems)
