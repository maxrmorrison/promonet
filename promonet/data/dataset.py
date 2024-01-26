import functools
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
        self.cache = promonet.CACHE_DIR / dataset
        self.partition = partition
        self.viterbi = '-viterbi' if promonet.VITERBI_DECODE_PITCH else ''

        # Get stems corresponding to partition
        partition_dict = promonet.load.partition(dataset)
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
                    promonet.AUGMENT_DIR / f'{dataset}-pitch.json'
                ) as file:
                    ratios = json.load(file)
                self.stems.extend([f'{stem}-p{ratios[stem]}' for stem in stems])
            if promonet.AUGMENT_LOUDNESS:
                with open(
                    promonet.AUGMENT_DIR / f'{dataset}-loudness.json'
                ) as file:
                    ratios = json.load(file)
                # TEMPORARY - some loudness files aren't in the cache
                self.stems.extend([
                    f'{stem}-l{ratios[stem]}' for stem in stems
                    if (self.cache / f'{stem}-l{ratios[stem]}.wav').exists()])

        # Omit files where the 50 Hz hum dominates the pitch estimation
        self.stems = [
            stem for stem in self.stems
            if (
                2 ** torch.log2(
                    torch.load(self.cache / f'{stem}{self.viterbi}-pitch.pt')
                ).mean()
            ) > 60.]

    def __getitem__(self, index):
        stem = self.stems[index]
        text = promonet.load.text(self.cache / f'{stem.split("-")[0]}.txt')
        audio = promonet.load.audio(
            self.cache / f'{stem}.wav').to(torch.float32)
        pitch = torch.load(
            self.cache / f'{stem}{self.viterbi}-pitch.pt').to(torch.float32)
        periodicity = torch.load(
            self.cache / f'{stem}{self.viterbi}-periodicity.pt'
        ).to(torch.float32)
        loudness = torch.load(
            self.cache / f'{stem}-loudness.pt').to(torch.float32)
        spectrogram = torch.load(
            self.cache / f'{stem}-spectrogram.pt').to(torch.float32)
        phonemes = promonet.load.ppg(
            self.cache / f'{stem}{ppgs.representation_file_extension()}',
            resample_length=spectrogram.shape[-1]).to(torch.float32)

        # Chunk during training
        if promonet.MODEL != 'vits' and self.partition.startswith('train'):
            frames = promonet.CHUNK_SIZE // promonet.HOPSIZE
            if audio.shape[1] < promonet.CHUNK_SIZE:
                audio = torch.nn.functional.pad(
                    audio,
                    (0, promonet.CHUNK_SIZE - audio.shape[1]),
                    mode='reflect')
                pad_frames = frames - pitch.shape[1]
                pad_fn = functools.partial(
                    torch.nn.functional.pad,
                    pad=(0, pad_frames),
                    mode='reflect')
                pitch = pad_fn(pitch)
                periodicity = pad_fn(periodicity)
                loudness = pad_fn(loudness)
                spectrogram = pad_fn(spectrogram)
                phonemes = pad_fn(phonemes)
            else:
                start_frame = torch.randint(pitch.shape[-1] - frames + 1, (1,)).item()
                start_sample = start_frame * promonet.HOPSIZE
                audio = audio[
                    :, start_sample:start_sample + promonet.CHUNK_SIZE]
                pitch = pitch[:, start_frame:start_frame + frames]
                periodicity = periodicity[:, start_frame:start_frame + frames]
                loudness = loudness[:, start_frame:start_frame + frames]
                spectrogram = spectrogram[:, start_frame:start_frame + frames]
                phonemes = phonemes[:, start_frame:start_frame + frames]

        # Get speaker index. Non-integer speaker names are assumed to be
        # for speaker adaptation and therefore default to index zero.
        if 'adapt' not in self.partition:
            speaker = int(stem.split('/')[0])
        else:
            speaker = 0

        # Data augmentation ratios
        augmentation = stem[-4:]
        if augmentation.startswith('-'):
            formant_ratio, loudness_ratio = 1., 1.
        elif augmentation.startswith('p'):
            formant_ratio = int(stem[-3:]) / 100.
            loudness_ratio = 1.
        elif augmentation.startswith('l'):
            formant_ratio = 1.
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
            formant_ratio,
            loudness_ratio,
            stem)

    def __len__(self):
        return len(self.stems)
