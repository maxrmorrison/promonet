import os
import random
from pathlib import Path

import numpy as np
import torch

import promovits


###############################################################################
# Datasets
###############################################################################


class PPGAudioSpeakerLoader(torch.utils.data.Dataset):

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.interp_method = hparams.interp_method

        random.seed(promovits.RANDOM_SEED)
        random.shuffle(self.audiopaths_sid_text)

        # Store spectrogram lengths for bucketing
        self.lengths = [
            os.path.getsize(path) // (2 * promovits.HOPSIZE)
            for path, _, _ in self.audiopaths_sid_text]

    def get_audio_ppg_speaker_pair(self, audiopath_sid_text):
        # Separate filenames and speaker_id
        audiopath, sid, text = audiopath_sid_text
        spec, wav = self.get_audio(audiopath)
        ppgpath = Path(audiopath).parent / f'{Path(audiopath).stem}-ppg.npy'
        ppg = self.get_ppg(ppgpath, spec.shape[1])
        sid = torch.LongTensor([int(sid)])
        return (ppg, spec, wav, sid)

    def get_audio(self, filename):
        audio = promovits.load.audio(filename)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = promovits.preprocess.spectrogram.from_audio(audio)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio

    def get_ppg(self, filename, length):
        """Load PPG features"""
        ppg = torch.from_numpy(np.load(filename))

        # Maybe resample length
        if ppg.shape[1] != length:
            ppg = torch.nn.functional.interpolate(
                ppg[None],
                size=length,
                mode=self.interp_method)[0]

        return ppg

    def __getitem__(self, index):
        return self.get_audio_ppg_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)

        random.seed(promovits.RANDOM_SEED)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * promovits.HOPSIZE))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return (text, spec, wav, sid)

    def get_audio(self, filename):
        audio = promovits.load.audio(filename)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = promovits.preprocess.spectrogram.from_audio(audio)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio

    def get_text(self, text):
        # TODO - should load as integers from .pt
        text_norm = cleaned_text_to_sequence(text)

        # Add blank tokens
        text_norm = commons.intersperse(text_norm, 0)

        return torch.LongTensor(text_norm)

    def get_sid(self, sid):
        return torch.LongTensor([int(sid)])

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)
