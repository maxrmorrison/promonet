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
        loudness,
        pitch,
        periodicity,
        phonemes,
        spectrograms,
        audio,
        speakers,
        spectral_balance_ratios,
        loudness_ratios,
        stems
    ) = zip(*batch)

    # Get lengths in samples
    lengths = torch.tensor([a.shape[1] for a in audio], dtype=torch.long)

    # Get batch indices sorted by length
    _, sorted_indices = torch.sort(lengths, dim=0, descending=True)

    # Get tensor size in frames and samples
    max_length_phonemes = max([p.shape[-1] for p in phonemes])
    max_length_samples = lengths.max().item()
    max_length_frames = promonet.convert.samples_to_frames(max_length_samples)

    # We store original lengths for, e.g., loss evaluation
    feature_lengths = torch.empty((len(batch),), dtype=torch.long)

    # Initialize padded tensors
    padded_phonemes = torch.zeros(
        (len(batch), promonet.PPG_CHANNELS, max_length_phonemes),
        dtype=torch.float)
    padded_pitch = torch.zeros(
        (len(batch), max_length_frames),
        dtype=torch.float)
    padded_periodicity = torch.zeros(
        (len(batch), max_length_frames),
        dtype=torch.float)
    padded_loudness = torch.zeros(
        (len(batch), promonet.NUM_FFT // 2 + 1, max_length_frames),
        dtype=torch.float)
    padded_spectrograms = torch.zeros(
        (len(batch), promonet.NUM_FFT // 2 + 1, max_length_frames),
        dtype=torch.float)
    padded_audio = torch.zeros(
        (len(batch), 1, max_length_samples),
        dtype=torch.float)
    for i, index in enumerate(sorted_indices):

        # Get lengths
        feature_lengths[i] = phonemes[index].shape[-1]

        # Prepare phoneme features
        padded_phonemes[i, :, :feature_lengths[i]] = phonemes[index]

        # Prepare prosody features
        padded_pitch[i, :feature_lengths[i]] = pitch[index]
        padded_periodicity[i, :feature_lengths[i]] = periodicity[index]
        padded_loudness[i, :, :feature_lengths[i]] = loudness[index]

        # Prepare spectrogram
        padded_spectrograms[i, :, :feature_lengths[i]] = \
            spectrograms[index]

        # Prepare audio
        padded_audio[i, :, :lengths[index]] = audio[index]

    # Collate speaker IDs or embeddings
    if promonet.ZERO_SHOT:
        speakers = torch.stack(speakers)
    else:
        speakers = torch.tensor(speakers, dtype=torch.long)

    # Sort stuff
    text = [text[i] for i in sorted_indices]
    stems = [stems[i] for i in sorted_indices]
    speakers = speakers[sorted_indices]
    spectral_balance_ratios = torch.tensor(
        spectral_balance_ratios, dtype=torch.float)[sorted_indices]
    loudness_ratios = torch.tensor(
        loudness_ratios, dtype=torch.float)[sorted_indices]

    return (
        text,
        padded_loudness,
        padded_pitch,
        padded_periodicity,
        padded_phonemes,
        speakers,
        spectral_balance_ratios,
        loudness_ratios,
        padded_spectrograms,
        padded_audio,
        stems)
