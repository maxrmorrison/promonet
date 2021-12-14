import torch

import promovits


###############################################################################
# Batch collation
###############################################################################


class Collate:
    """Collate from phonemes, spectrograms, audio, and speaker identities"""

    def __getitem__(batch):
        batch_size = len(batch)

        # Unpack
        (
            phonemes,
            pitch,
            periodicity,
            loudness,
            spectrograms,
            audio,
            speakers
         ) = zip(*batch)

        # Get lengths in samples
        lengths = torch.tensor([a.shape[1] for a in audio], dtype=torch.long)

        # Get batch indices sorted by length
        _, sorted_indices = torch.sort(lengths, dim=0, descending=True)

        # Get tensor size in frames and samples
        max_length_phonemes = max([len(phoneme) for phoneme in phonemes])
        max_length_samples = lengths.max()
        max_length_frames = max_length_samples // promovits.HOPSIZE

        # We store original lengths for, e.g., loss evaluation
        phoneme_lengths = torch.tensor((batch_size,), dtype=torch.long)
        spectrogram_lengths = torch.tensor((batch_size,), dtype=torch.long)
        audio_lengths = torch.tensor((batch_size,), dtype=torch.long)

        # Initialize padded tensors
        if promovits.PPG_FEATURES:
            # TODO - split pitch features into a long tensor
            # TODO - periodicity and loudness features
            padded_phonemes = torch.zeros(
                (len(batch), promovits.NUM_FEATURES, max_length_phonemes),
                dtype=torch.float)
        else:
            padded_phonemes = torch.zeros(
                (batch_size, max_length_phonemes),
                dtype=torch.long)
        padded_spectrograms = torch.zeros(
            (batch_size, promovits.NUM_FFT // 2 + 1, max_length_frames),
            dtype=torch.float)
        padded_audio = torch.zeros(
            (batch_size, 1, max_length_samples),
            dtype=torch.float)
        for i, index in enumerate(sorted_indices):

            # Get lengths
            phoneme_lengths[i] = len(phonemes[index])
            spectrogram_lengths[i] = lengths[index] // promovits.HOPSIZE
            audio_lengths[i] = lengths[index]

            # Place in padded tensor
            padded_phonemes[i, :phoneme_lengths[i]] = phonemes[index]
            padded_spectrograms[i, :, :spectrogram_lengths[i]] = \
                spectrograms[index]
            padded_audio[i, :, :audio_lengths[i]] = audio[index]

        return (
            padded_phonemes,
            phoneme_lengths,
            padded_spectrograms,
            spectrogram_lengths,
            padded_audio,
            audio_lengths,
            torch.tensor(speakers, dtype=torch.long))
