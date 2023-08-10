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
        speakers,
        ratios,
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
    spectrogram_lengths = torch.empty((len(batch),), dtype=torch.long)

    # Initialize padded tensors
    if promonet.MODEL == 'vits':
        padded_phonemes = torch.zeros(
            (len(batch), max_length_phonemes),
            dtype=torch.long)
    else:
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
        (len(batch), max_length_frames),
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
        spectrogram_lengths[i] = promonet.convert.samples_to_frames(
            lengths[index].item())

        # Prepare phoneme features
        padded_phonemes[i, ..., :feature_lengths[i]] = phonemes[index]

        # Prepare prosody features
        padded_pitch[i, :spectrogram_lengths[i]] = pitch[index]
        padded_periodicity[i, :spectrogram_lengths[i]] = periodicity[index]
        padded_loudness[i, :spectrogram_lengths[i]] = loudness[index]

        # Prepare spectrogram
        padded_spectrograms[i, :, :spectrogram_lengths[i]] = \
            spectrograms[index]

        # Prepare audio
        padded_audio[i, :, :lengths[index]] = audio[index]

    # Sort stuff
    text = [text[i] for i in sorted_indices]
    stems = [stems[i] for i in sorted_indices]
    speakers = torch.tensor(speakers, dtype=torch.long)[sorted_indices]
    ratios = torch.tensor(ratios, dtype=torch.float)[sorted_indices]

    return (
        text,
        padded_phonemes,
        padded_pitch,
        padded_periodicity,
        padded_loudness,
        feature_lengths,
        speakers,
        ratios,
        padded_spectrograms,
        spectrogram_lengths,
        padded_audio,
        stems)
