import torch

import promonet


###############################################################################
# Batch collation
###############################################################################


def collate(batch):
    """Collate from features, spectrograms, audio, and speaker identities"""
    # Unpack
    (
        text,
        phonemes,
        pitch,
        periodicity,
        loudness,
        spectrograms,
        audio,
        templates,
        speakers,
        ratios
    ) = zip(*batch)

    # Get lengths in samples
    lengths = torch.tensor([a.shape[1] for a in audio], dtype=torch.long)

    # Get batch indices sorted by length
    _, sorted_indices = torch.sort(lengths, dim=0, descending=True)

    # Get tensor size in frames and samples
    max_length_phonemes = max([p.shape[-1] for p in phonemes])
    max_length_samples = lengths.max().item()
    max_length_frames = max_length_samples // promonet.HOPSIZE

    # We store original lengths for, e.g., loss evaluation
    feature_lengths = torch.empty((len(batch),), dtype=torch.long)
    spectrogram_lengths = torch.empty((len(batch),), dtype=torch.long)

    # Initialize padded tensors
    if promonet.PPG_FEATURES or promonet.SPECTROGRAM_ONLY:
        padded_phonemes = torch.zeros(
            (len(batch), promonet.PPG_CHANNELS, max_length_phonemes),
            dtype=torch.float)
    else:
        padded_phonemes = torch.zeros(
            (len(batch), max_length_phonemes),
            dtype=torch.long)
    padded_pitch = torch.zeros(
        (len(batch), max_length_frames),
        dtype=torch.float)
    padded_periodicity = torch.zeros(
        (len(batch), max_length_frames),
        dtype=torch.float)
    padded_loudness = torch.zeros(
        (len(batch), max_length_frames),
        dtype=torch.float)
    padded_spectrograms = torch.zeros(
        (len(batch), promonet.NUM_FFT // 2 + 1, max_length_frames),
        dtype=torch.float)
    padded_audio = torch.zeros(
        (len(batch), 1, max_length_samples),
        dtype=torch.float)
    padded_templates = torch.zeros(
        (len(batch), 1, max_length_frames * promonet.HOPSIZE),
        dtype=torch.float)
    for i, index in enumerate(sorted_indices):

        # Get lengths
        feature_lengths[i] = phonemes[index].shape[-1]
        spectrogram_lengths[i] = lengths[index].item() // promonet.HOPSIZE

        # Prepare phoneme features
        if promonet.PPG_FEATURES or promonet.SPECTROGRAM_ONLY:
            padded_phonemes[i, :, :feature_lengths[i]] = phonemes[index]
        else:
            padded_phonemes[i, :feature_lengths[i]] = phonemes[index]

        # Prepare prosody features
        padded_pitch[i, :spectrogram_lengths[i]] = pitch[index]
        padded_periodicity[i, :spectrogram_lengths[i]] = periodicity[index]
        padded_loudness[i, :spectrogram_lengths[i]] = loudness[index]

        # Prepare spectrogram
        padded_spectrograms[i, :, :spectrogram_lengths[i]] = \
            spectrograms[index]

        # Prepare audio
        padded_audio[i, :, :lengths[index]] = audio[index]

        # Prepare template
        length = spectrogram_lengths[i] * promonet.HOPSIZE
        padded_templates[i, :, :length] = templates[index]

    return (
        text,
        padded_phonemes,
        padded_pitch,
        padded_periodicity,
        padded_loudness,
        feature_lengths,
        torch.tensor(speakers, dtype=torch.long),
        torch.tensor(ratios, dtype=torch.float),
        padded_spectrograms,
        spectrogram_lengths,
        padded_audio,
        padded_templates)
