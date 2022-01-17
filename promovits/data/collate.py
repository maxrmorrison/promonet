import torch

import promovits


###############################################################################
# Batch collation
###############################################################################


def collate(batch):
    """Collate from features, spectrograms, audio, and speaker identities"""
    # Unpack
    (
        features,
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
    max_length_features = max([feature.shape[-1] for feature in features])
    max_length_samples = lengths.max().item()
    max_length_frames = max_length_samples // promovits.HOPSIZE

    # We store original lengths for, e.g., loss evaluation
    feature_lengths = torch.empty((len(batch),), dtype=torch.long)
    spectrogram_lengths = torch.empty((len(batch),), dtype=torch.long)

    # Initialize padded tensors
    if promovits.PPG_FEATURES:
        num_float_features = \
            promovits.NUM_FEATURES - \
            promovits.PITCH_FEATURES * \
            promovits.PITCH_EMBEDDING_SIZE
        padded_features = torch.zeros(
            (len(batch), num_float_features, max_length_features),
            dtype=torch.float)
    else:
        padded_features = torch.zeros(
            (len(batch), max_length_features),
            dtype=torch.long)
    padded_pitch = torch.zeros(
        (len(batch), max_length_features),
        dtype=torch.long)
    padded_spectrograms = torch.zeros(
        (len(batch), promovits.NUM_FFT // 2 + 1, max_length_frames),
        dtype=torch.float)
    padded_audio = torch.zeros(
        (len(batch), 1, max_length_samples),
        dtype=torch.float)
    for i, index in enumerate(sorted_indices):

        # Get lengths
        feature_lengths[i] = features[index].shape[-1]
        spectrogram_lengths[i] = lengths[index].item() // promovits.HOPSIZE

        # Prepare pitch, periodicity, loudness, and features
        if promovits.PPG_FEATURES:
            j = promovits.PPG_CHANNELS
            padded_features[i, :j, :feature_lengths[i]] = features[index]
            if promovits.LOUDNESS_FEATURES:
                padded_features[i, j, :feature_lengths[i]] = loudness[index]
                j += 1
            if promovits.PERIODICITY_FEATURES:
                padded_features[i, j, :feature_lengths[i]] = periodicity[index]
            padded_pitch[i, :feature_lengths[i]] = pitch[index]

        else:
            padded_features[i, :feature_lengths[i]] = features[index]

        # Prepare spectrogram
        padded_spectrograms[i, :, :spectrogram_lengths[i]] = \
            spectrograms[index]

        # Prepare audio
        padded_audio[i, :, :lengths[index]] = audio[index]

    return (
        padded_features,
        feature_lengths,
        padded_pitch,
        torch.tensor(speakers, dtype=torch.long),
        padded_spectrograms,
        spectrogram_lengths,
        padded_audio)
