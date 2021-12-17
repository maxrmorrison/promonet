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
        audio = promovits.load.audio(self.cache / f'{stem}.wav')
        pitch = self.get_pitch(stem)
        periodicity = torch.load(self.cache / f'{stem}-periodicity.pt')
        loudness = torch.load(self.cache / f'{stem}-loudness.pt')
        speaker = torch.tensor(int(stem.split('-')[0]), dtype=torch.long)
        spectrogram = torch.load(self.cache / f'{stem}-spectrogram.pt')

        # Load supervised or unsupervised phoneme features
        if promovits.PPG_FEATURES:
            phonemes = self.get_ppg(stem, spectrogram.shape[1])
        else:
            phonemes = promovits.load.phonemes(
                self.cache / f'{stem}-phonemes.pt')

        return (
            phonemes,
            pitch,
            periodicity,
            loudness,
            spectrogram,
            audio,
            speaker)

    def __len__(self):
        return len(self.stems)

    def get_pitch(self, stem):
        """Load pitch features"""
        pitch = torch.load(self.cache / f'{stem}-pitch.pt')
        pitch[pitch < promovits.FMIN] = promovits.FMIN
        pitch[pitch > promovits.FMAX] = promovits.FMAX
        return promovits.convert.hz_to_bins(pitch)


    def get_ppg(self, stem, length):
        """Load PPG features"""
        ppg = torch.load(self.cache / f'{stem}-ppg.pt')

        # Maybe resample length
        if ppg.shape[1] != length:
            ppg = torch.nn.functional.interpolate(
                ppg[None],
                size=length,
                mode=promovits.PPG_INTERP_METHOD)[0]

        return ppg

    @functools.cached_property
    def speakers(self):
        """Retrieve the list of speaker ids"""
        return sorted(list(set(stem.split('-')[0] for stem in self.stems)))
