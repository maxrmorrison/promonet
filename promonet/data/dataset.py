import functools
import json
import random

import torch

import promonet
import ppgs


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, partition, adapt=promonet.ADAPTATION):
        super().__init__()
        self.cache = promonet.CACHE_DIR / dataset
        self.partition = partition
        self.viterbi = '-viterbi' if promonet.VITERBI_DECODE_PITCH else ''

        # Get stems corresponding to partition
        partition_dict = promonet.load.partition(dataset, adapt)
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
        self.speaker_stems = {}
        for stem in self.stems:
            speaker = stem.split('/')[0]
            if speaker not in self.speaker_stems:
                self.speaker_stems[speaker] = [stem]
            else:
                self.speaker_stems[speaker].append(stem)

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
        spectrogram = torch.load(
            self.cache / f'{stem}-spectrogram.pt').to(torch.float32)
        phonemes = promonet.load.ppg(
            self.cache / f'{stem}{ppgs.representation_file_extension()}',
            resample_length=spectrogram.shape[-1]
        ).to(torch.float32)

        # For loudness augmentation, use original loudness to disentangle
        if stem.split('-')[-1].startswith('l'):
            loudness_file = self.cache / f'{stem[:-4]}100-loudness.pt'
        else:
            loudness_file = self.cache / f'{stem}-loudness.pt'
        loudness = torch.load(loudness_file).to(torch.float32)

        # Chunk during training
        if self.partition.startswith('train'):
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

        if promonet.ZERO_SHOT:

            # Load speaker embedding
            if promonet.ZERO_SHOT_SHUFFLE and 'train' in self.partition:
                random_speaker_stem = stem
                while random_speaker_stem == stem:
                    random_speaker_stem = random.choice(self.speaker_stems[stem.split('/')[0]])
                speaker = torch.load(self.cache / f'{random_speaker_stem}-speaker.pt')
            else:
                speaker = torch.load(self.cache / f'{stem}-speaker.pt')

        else:

            # Get speaker index. Non-integer speaker names are assumed to be
            # for speaker adaptation and therefore default to index zero.
            if 'adapt' not in self.partition:
                speaker = int(stem.split('/')[0])
            else:
                speaker = 0
            speaker = torch.tensor(speaker, dtype=torch.long)

        # Data augmentation ratios
        augmentation = stem[-4:]
        if augmentation.startswith('-'):
            spectral_balance_ratios, loudness_ratio = 1., 1.
        elif augmentation.startswith('p'):
            spectral_balance_ratios = int(stem[-3:]) / 100.
            loudness_ratio = 1.
        elif augmentation.startswith('l'):
            spectral_balance_ratios = 1.
            loudness_ratio = int(stem[-3:]) / 100.
        else:
            raise ValueError(
                f'Unrecognized augmentation string {augmentation}')

        return (
            text,
            loudness,
            pitch,
            periodicity,
            phonemes,
            spectrogram,
            audio,
            speaker,
            spectral_balance_ratios,
            loudness_ratio,
            stem)

    def __len__(self):
        return len(self.stems)
