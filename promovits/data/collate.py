import torch

import promovits


###############################################################################
# Batch collation
###############################################################################


# TODO - merge collate functions
class PPGCollate():

    def __call__(self, batch):
        """Collate batch from ppg, spectrograms, waveforms, and speaker ids"""
        batch_size = len(batch)

        # Unpack
        ppgs, spectrograms, audio, speakers = zip(*batch)

        # Get lengths in samples
        lengths = torch.tensor([a.shape[1] for a in audio], dtype=torch.long)

        # Get batch indices sorted by length
        _, sorted_indices = torch.sort(lengths, dim=0, descending=True)

        # Get tensor size in frames and samples
        max_length_samples = lengths.max()
        max_length_frames = max_length_samples // promovits.HOPSIZE

        # We store original lengths for, e.g., loss evaluation
        ppg_lengths = torch.zeros((batch_size,), dtype=torch.long)
        spectrogram_lengths = torch.zeros((batch_size,), dtype=torch.long)
        audio_lengths = torch.zeros((batch_size,), dtype=torch.long)

        # Initialize padded tensors
        padded_ppgs = torch.zeros(
            (len(batch), promovits.PPG_CHANNELS, max_length_frames),
            dtype=torch.float)
        padded_spectrograms = torch.zeros(
            (len(batch), promovits.NUM_FFT // 2 + 1, max_length_frames),
            dtype=torch.float)
        padded_audio = torch.zeros(
            (len(batch), 1, max_length_samples),
            dtype=torch.float)
        for i, index in enumerate(sorted_indices):

            # Get lengths
            ppg_lengths[i] = lengths[index] // promovits.HOPSIZE
            spectrogram_lengths[i] = lengths[index] // promovits.HOPSIZE
            audio_lengths[i] =  lengths[index]

            # Place in padded tensor
            padded_ppgs[i, :, :ppg_lengths[i]] = ppgs[index]
            padded_spectrograms[i, :, :spectrogram_lengths[i]] = \
                spectrograms[index]
            padded_audio[i, :, :audio_lengths[i]] = audio[index]

        return (
            padded_ppgs,
            ppg_lengths,
            padded_spectrograms,
            spectrogram_lengths,
            padded_audio,
            audio_lengths,
            torch.tensor(speakers, dtype=torch.long))


class TextCollate():

    def __call__(self, batch):
        """Collate from text, spectrograms, audio, and speaker identities"""
        batch_size = len(batch)

        # Unpack
        texts, spectrograms, audio, speakers = zip(*batch)

        # Get lengths in samples
        lengths = torch.tensor([a.shape[1] for a in audio], dtype=torch.long)

        # Get batch indices sorted by length
        _, sorted_indices = torch.sort(lengths, dim=0, descending=True)

        # Get tensor size in frames and samples
        max_length_text = max([len(text) for text in texts])
        max_length_samples = lengths.max()
        max_length_frames = max_length_samples // promovits.HOPSIZE

        # We store original lengths for, e.g., loss evaluation
        text_lengths = torch.tensor((batch_size,), dtype=torch.long)
        spectrogram_lengths = torch.tensor((batch_size,), dtype=torch.long)
        audio_lengths = torch.tensor((batch_size,), dtype=torch.long)

        # Initialize padded tensors
        padded_text = torch.zeros(
            (batch_size, max_length_text),
            dtype=torch.long)
        padded_spectrograms = torch.zeros(
            (batch_size, promovits.NUM_FFT // 2 + 1, max_length_frames),
            dtype=torch.float)
        padded_audio = torch.zeros(
            (batch_size, 1, max_length_samples),
            dtype=torch.float)
        for i, index in enumerate(sorted_indices):

            # Get lengths
            text_lengths[i] = len(texts[index])
            spectrogram_lengths[i] = lengths[index] // promovits.HOPSIZE
            audio_lengths[i] = lengths[index]

            # Place in padded tensor
            padded_text[i, :text_lengths[i]] = texts[index]
            padded_spectrograms[i, :, :spectrogram_lengths[i]] = \
                spectrograms[index]
            padded_audio[i, :, :audio_lengths[i]] = audio[index]

        return (
            padded_text,
            text_lengths,
            padded_spectrograms,
            spectrogram_lengths,
            padded_audio,
            audio_lengths,
            torch.tensor(speakers, dtype=torch.long))
