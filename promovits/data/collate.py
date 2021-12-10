import torch

import promovits


###############################################################################
# Batch collation
###############################################################################


class PPGCollate():

    def __call__(self, batch):
        """Collate batch from ppg, spectrograms, waveforms, and speaker ids"""
        batch_size = len(batch)

        # Unpack
        ppgs, spectrograms, audio, speakers = zip(*batch)

        # Get lengths in frames
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
            padded_spectrograms[i, :, :spectrogram_lengths[i]] = spectrograms[index]
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
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spectrogram_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        padded_spectrograms = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        padded_audio = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        padded_spectrograms.zero_()
        padded_audio.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            padded_spectrograms[i, :, :spec.size(1)] = spec
            spectrogram_lengths[i] = spec.size(1)

            wav = row[2]
            padded_audio[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

        return text_padded, text_lengths, spec_padded, spectrogram_lengths, padded_audio, wav_lengths, sid
