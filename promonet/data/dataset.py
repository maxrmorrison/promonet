import functools
import json
import os

import torch

import promonet


###############################################################################
# Speech modification dataset
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

        # Store spectrogram lengths for bucketing
        audio_files = list([self.cache / f'{stem}.wav' for stem in self.stems])
        self.spectrogram_lengths = [
            os.path.getsize(audio_file) // (2 * promonet.HOPSIZE)
            for audio_file in audio_files]

    def __getitem__(self, index):
        stem = self.stems[index]
        text = promonet.load.text(self.cache / f'{stem[:-4]}.txt')
        audio = promonet.load.audio(self.cache / f'{stem}.wav')
        pitch = promonet.load.pitch(self.cache / f'{stem}-pitch.pt')
        periodicity = torch.load(self.cache / f'{stem}-periodicity.pt')
        loudness = torch.load(self.cache / f'{stem}-loudness.pt')
        spectrogram = torch.load(self.cache / f'{stem}-spectrogram.pt')

        # Get speaker index. Non-integer speaker names are assumed to be
        # for speaker adaptation and therefore default to index zero.
        if 'adapt' not in self.partition:
            try:
                speaker = int(stem.split('/')[0])
            except ValueError:
                speaker = 0
        else:
            speaker = 0

        # Load supervised or unsupervised phoneme features
        if promonet.PPG_FEATURES or promonet.SPECTROGRAM_ONLY:
            phonemes = self.get_ppg(stem, spectrogram.shape[1])
        else:
            phonemes = promonet.load.phonemes(
                self.cache / f'{stem}-text.pt')

        return (
            text,
            phonemes,
            pitch,
            periodicity,
            loudness,
            spectrogram,
            audio,
            torch.tensor(speaker, dtype=torch.long),
            int(stem[-3:]) / 100.)

    def __len__(self):
        return len(self.stems)

    def get_ppg(self, stem, length):
        """Load PPG features"""
        ppg = torch.load(self.cache / f'{stem}-ppg.pt')

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
