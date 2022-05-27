import functools
import os

import torch

import promovits


###############################################################################
# Speech modification dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, partition):
        super().__init__()
        self.stems = promovits.load.partition(dataset)[partition]
        self.cache = promovits.CACHE_DIR / dataset

        # Store spectrogram lengths for bucketing
        audio_files = list([self.cache / f'{stem}.wav' for stem in self.stems])
        self.spectrogram_lengths = [
            os.path.getsize(audio_file) // (2 * promovits.HOPSIZE)
            for audio_file in audio_files]

    def __getitem__(self, index):
        stem = self.stems[index]
        text = promovits.load.text(self.cache / f'{stem}.txt')
        audio = promovits.load.audio(self.cache / f'{stem}.wav')
        pitch = promovits.load.pitch(self.cache / f'{stem}-pitch.pt')
        periodicity = torch.load(self.cache / f'{stem}-periodicity.pt')
        loudness = torch.load(self.cache / f'{stem}-loudness.pt')
        spectrogram = torch.load(self.cache / f'{stem}-spectrogram.pt')

        # Get speaker index. Non-integer speaker names are assumed to be
        # for speaker adaptation and therefore default to index zero.
        try:
            speaker = int(stem.split('/')[0])
        except ValueError:
            speaker = 0

        # Load supervised or unsupervised phoneme features
        if promovits.PPG_FEATURES:
            phonemes = self.get_ppg(stem, spectrogram.shape[1])
        else:
            phonemes = promovits.load.phonemes(
                self.cache / f'{stem}-text.pt')

        return (
            text,
            phonemes,
            pitch,
            periodicity,
            loudness,
            spectrogram,
            audio,
            torch.tensor(speaker, dtype=torch.long))

    def __len__(self):
        return len(self.stems)

    def get_ppg(self, stem, length):
        """Load PPG features"""
        ppg = torch.load(self.cache / f'{stem}-ppg.pt')

        # Maybe resample length
        if ppg.shape[1] != length:
            mode = promovits.PPG_INTERP_METHOD
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
