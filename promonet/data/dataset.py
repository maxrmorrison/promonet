import functools
import json

import numpy as np
import penn
import pysodic
import torch
import torchaudio
import os

import promonet
import ppgs




###############################################################################
# Metadata
###############################################################################

class Metadata:

    def __init__(self, dataset, partition, overwrite_cache=False):
        self.name = dataset
        self.cache = promonet.CACHE_DIR / self.name
        stems = [stem for stem in promonet.load.partition(self.name)[partition]]
        metadata_file = self.cache / f'{partition}-metadata.json'

        # Get data stems assuming no augmentation
        self.stems = [f'{stem}-100' for stem in stems]

        # For training, maybe add augmented data
        # This also applies to adaptation partitions: train-adapt-xx
        if 'train' in partition and promonet.AUGMENT_PITCH:
            with open(promonet.AUGMENT_DIR / f'{dataset}.json') as file:
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
                    len(
                        promonet.load.phonemes(
                            self.cache / f'{stem}-phonemes.pt')
                    ) < promonet.MAX_TEXT_LENGTH and
                    promonet.convert.samples_to_frames(
                        torchaudio.info(self.cache / f'{stem}.wav').num_frames
                    ) < promonet.MAX_FRAME_LENGTH
                )
            ]

        if overwrite_cache and metadata_file.exists():
            print('overwriting metadata cache')
            os.remove(metadata_file)
        if metadata_file.exists():
            print('using cached metadata')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            print('generating metadata from scratch')
            metadata = {}
        print('preparing dataset metadata (operation may be slow)')
        self.lengths = []
        for stem in self.stems:
            try:
                self.lengths.append(metadata[stem])
            except KeyError:
                length = promonet.convert.samples_to_frames(
                    torchaudio.info(self.cache / f'{stem}.wav').num_frames)
                metadata[stem] = length
                self.lengths.append(length)
        with open(metadata_file, 'w+') as f:
            json.dump(metadata, f)

    def __len__(self):
        return len(self.stems)

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
        pitch = promonet.load.pitch(self.cache / f'{stem}-pitch.pt')
        periodicity = torch.load(self.cache / f'{stem}-periodicity.pt')
        loudness = torch.load(self.cache / f'{stem}-loudness.pt')
        spectrogram = torch.load(self.cache / f'{stem}-spectrogram.pt')

        # Apply linear interpolation to unvoiced pitch regions
        pitch = penn.voicing.interpolate(
            pitch,
            periodicity,
            pysodic.DEFAULT_VOICING_THRESHOLD)

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
            phonemes = self.get_ppg(stem, spectrogram.shape[1])

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

    def get_ppg(self, stem, length):
        """Load PPG features"""
        feature = 'ppg'

        # Maybe use a different type of PPGs
        if promonet.PPG_MODEL is not None:
            feature = promonet.PPG_MODEL

        ppg = torch.load(self.cache / f'{stem}-{feature}.pt')
        if promonet.PPG_MODEL is not None and 'ppg' not in promonet.PPG_MODEL and ppgs.FRONTEND is not None:
            if not hasattr(self.get_ppg, 'frontend'):
                self.get_ppg.__func__.frontend = ppgs.FRONTEND()
            with torch.no_grad():
                ppg = self.get_ppg.__func__.frontend(ppg[None]).squeeze(dim=0)

        # Maybe resample length
        if ppg.shape[1] != length:
            mode = promonet.PPG_INTERP_METHOD
            #TODO shperical linear
            ppg = torch.nn.functional.interpolate(
                ppg[None].to(torch.float32),
                size=length,
                mode=mode,
                align_corners=None if mode == 'nearest' else False)[0]

        return ppg

    @functools.cached_property
    def speakers(self):
        """Retrieve the list of speaker ids"""
        return sorted(list(self.cache.glob('*')))


###############################################################################
# Metadata
###############################################################################

class Metadata:

    def __init__(self, name, partition, overwrite_cache=False):
        self.name = name
        self.cache = promonet.CACHE_DIR / self.name

        # Load and possibly augment stems
        stems = promonet.load.partition(self.name)[partition]
        self.stems = [f'{stem}-100' for stem in stems]

        # For training, maybe add augmented data
        # This also applies to adaptation partitions: train-adapt-xx
        if 'train' in partition and promonet.AUGMENT_PITCH:
            with open(promonet.AUGMENT_DIR / f'{name}.json') as file:
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
                    len(
                        promonet.load.phonemes(
                            self.cache / f'{stem}-phonemes.pt')
                    ) < promonet.MAX_TEXT_LENGTH and
                    promonet.convert.samples_to_frames(
                        torchaudio.info(self.cache / f'{stem}.wav').num_frames
                    ) < promonet.MAX_FRAME_LENGTH
                )
            ]

        metadata_file = self.cache / f'{partition}-metadata.json'
        if overwrite_cache and metadata_file.exists():
            os.remove(metadata_file)
        if metadata_file.exists():
            print('using cached metadata')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            print('generating metadata from scratch')
            metadata = {}
        print('preparing dataset metadata (operation may be slow)')
        self.lengths = []
        for stem in self.stems:
            try:
                self.lengths.append(metadata[stem])
            except KeyError:
                length = torchaudio.info(self.cache / (stem + '.wav')).num_frames // promonet.HOPSIZE
                metadata[stem] = length
                self.lengths.append(length)
        with open(metadata_file, 'w+') as f:
            json.dump(metadata, f)

    def __len__(self):
        return len(self.stems)