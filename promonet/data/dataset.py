import functools
import json

import numpy as np
import penn
import pysodic
import torch
import torchaudio

import promonet


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, partition):
        super().__init__()
        self.partition = partition
        self.cache = promonet.CACHE_DIR / dataset

        # Get data stems assuming no augmentation
        stems = promonet.load.partition(dataset)[partition]
        self.stems = [f'{stem}-100' for stem in stems]

        # For training, maybe add augmented data
        # This also applies to adaptation partitions: train-adapt-xx
        if 'train' in partition and promonet.AUGMENT_PITCH:
            with open(promonet.AUGMENT_DIR / f'{dataset}.json') as file:
                ratios = json.load(file)
            self.stems.extend([f'{stem}-{ratios[stem]}' for stem in stems])

        # Maybe limit the maximum length of text during training to improve
        # GPU utilization
        if (
            ('train' in partition or 'valid' in partition) and
            promonet.MAX_TEXT_LENGTH is not None
        ):
            self.stems = [
                stem for stem in self.stems if len(
                    promonet.load.phonemes(
                        self.cache / f'{stem}-phonemes.pt')
                ) < promonet.MAX_TEXT_LENGTH]

        # Store spectrogram lengths for bucketing
        self.lengths = [
            promonet.convert.samples_to_frames(
                torchaudio.info(self.cache / f'{stem}.wav').num_frames)
            for stem in self.stems]

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
        if promonet.PPG_FEATURES or promonet.SPECTROGRAM_ONLY:
            phonemes = self.get_ppg(stem, spectrogram.shape[1])
        else:
            phonemes = promonet.load.phonemes(
                self.cache / f'{stem}-phonemes.pt',
                interleave=True)

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
            feature += '-' + promonet.PPG_MODEL

        ppg = torch.load(self.cache / f'{stem}-{feature}.pt')

        # Maybe resample length
        if ppg.shape[1] != length:
            mode = promonet.PPG_INTERP_METHOD
            ppg = torch.nn.functional.interpolate(
                ppg[None],
                size=length,
                mode=mode,
                align_corners=None if mode == 'nearest' else False)[0]

        return ppg

    @functools.cached_property
    def speakers(self):
        """Retrieve the list of speaker ids"""
        return sorted(list(self.cache.glob('*')))
