import math

import torch

import promonet


###############################################################################
# Model definition
###############################################################################


class Generator(torch.nn.Module):

    def __init__(self, n_speakers=109):
        super().__init__()
        self.n_speakers = n_speakers

        # Input feature encoding
        if promonet.MODEL in ['end-to-end', 'two-stage', 'vits']:
            self.prior_encoder = PriorEncoder()

        # Phoneme duration predictor
        if promonet.MODEL == 'vits':
            self.dp = promonet.model.DurationPredictor(
                promonet.HIDDEN_CHANNELS,
                192,
                0.5,
                4)

        # Vocoder
        self.vocoder = promonet.model.get_vocoder(
            promonet.LATENT_FEATURES,
            promonet.GLOBAL_CHANNELS)

        # Latent flow
        if promonet.MODEL in ['end-to-end', 'vits']:

            # Acoustic encoder
            self.posterior_encoder = PosteriorEncoder()

            # Normalizing flow
            self.flow = promonet.model.flow.Block(
                promonet.HIDDEN_CHANNELS,
                promonet.HIDDEN_CHANNELS,
                5,
                1,
                4,
                gin_channels=promonet.GLOBAL_CHANNELS)

        # Speaker embedding
        self.speaker_embedding = torch.nn.Embedding(
            n_speakers,
            promonet.SPEAKER_CHANNELS)

        # Separate speaker embeddings in two-stage models
        if promonet.MODEL == 'two-stage':
            self.speaker_embedding_vocoder = torch.nn.Embedding(
                n_speakers,
                promonet.SPEAKER_CHANNELS)

        # Pitch embedding
        if 'pitch' in promonet.INPUT_FEATURES and promonet.PITCH_EMBEDDING:
            self.pitch_embedding = torch.nn.Embedding(
                promonet.PITCH_BINS,
                promonet.PITCH_EMBEDDING_SIZE)

    def forward(
        self,
        phonemes,
        pitch,
        periodicity,
        loudness,
        lengths,
        speakers,
        ratios=None,
        spectrograms=None,
        spectrogram_lengths=None):
        # Default augmentation ratio is 1
        if ratios is None and promonet.AUGMENT_PITCH:
            ratios = torch.ones(
                1 if phonemes.dim() == 2 else len(phonemes),
                dtype=torch.float,
                device=phonemes.device)

        # Get latent representation
        (
            latents,
            speaker_embeddings,
            mask,
            slice_indices,
            spectrogram_slice,
            *args
         ) = self.latents(
            phonemes,
            pitch,
            periodicity,
            loudness,
            lengths,
            speakers,
            ratios,
            spectrograms,
            spectrogram_lengths
        )

        if promonet.MODEL == 'two-stage':

            # Use different speaker embedding for two-stage models
            speaker_embeddings = self.speaker_embedding_vocoder(
                speakers).unsqueeze(-1)

            # Maybe add augmentation ratios
            if ('pitch' in promonet.INPUT_FEATURES) and promonet.AUGMENT_PITCH:
                speaker_embeddings = torch.cat(
                    (speaker_embeddings, ratios[:, None, None]),
                    dim=1)

            # During first stage of two-stage generation, train the vocoder on
            # ground-truth spectrograms
            if self.training:
                if promonet.TWO_STAGE_1:
                    spectrogram_slice, latents = (
                        latents,
                        promonet.preprocess.spectrogram.linear_to_mel(
                            spectrogram_slice)
                    )
                else:
                    spectrogram_slice = latents

        else:

            spectrogram_slice = None

        # Decode latent representation to waveform
        generated = self.vocoder(latents, g=speaker_embeddings)

        return generated, mask, slice_indices, spectrogram_slice, *args

    def duration_prediction(
        self,
        predicted_mean,
        predicted_logstd,
        embeddings,
        speaker_embeddings,
        feature_mask,
        mask=None,
        prior=None):
        """Predict phoneme durations"""
        if promonet.MODEL == 'vits':

            if self.training:

                # Compute attention mask
                attention_mask = feature_mask * mask.permute(0, 2, 1)

                # Compute monotonic alignment
                with torch.no_grad():
                    s_p_sq_r = torch.exp(-2 * predicted_logstd)
                    neg_cent1 = torch.sum(
                        -0.5 * math.log(2 * math.pi) - predicted_logstd,
                        [1],
                        keepdim=True)
                    neg_cent2 = torch.matmul(
                        -0.5 * (prior ** 2).transpose(1, 2),
                        s_p_sq_r)
                    neg_cent3 = torch.matmul(
                        prior.transpose(1, 2),
                        predicted_mean * s_p_sq_r)
                    neg_cent4 = torch.sum(
                        -0.5 * (predicted_mean ** 2) * s_p_sq_r,
                        [1],
                        keepdim=True)
                    neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
                    attention = promonet.model.align.maximum_path(
                        neg_cent,
                        attention_mask.squeeze(1)
                    ).unsqueeze(1).detach()

                # Calcuate duration of each feature
                w = attention.sum(2)

                # Predict durations from text
                durations = self.dp(
                    embeddings,
                    feature_mask,
                    w,
                    g=speaker_embeddings)
                durations = durations / torch.sum(feature_mask)

                # Expand sequence using predicted durations
                predicted_mean = torch.matmul(
                    attention.squeeze(1),
                    predicted_mean.transpose(1, 2)).transpose(1, 2)
                predicted_logstd = torch.matmul(
                    attention.squeeze(1),
                    predicted_logstd.transpose(1, 2)).transpose(1, 2)

            else:

                # Predict durations from text
                logw = self.dp(
                    embeddings,
                    feature_mask,
                    g=speaker_embeddings,
                    reverse=True,
                    noise_scale=promonet.NOISE_SCALE_W_INFERENCE)
                w = torch.exp(logw) * feature_mask
                durations = torch.ceil(w)

                # Get new frame resolution mask
                y_lengths = torch.clamp_min(torch.sum(durations, [1, 2]), 1).long()
                mask = torch.unsqueeze(
                    sequence_mask(y_lengths),
                    1
                ).to(feature_mask.dtype)

                # Compute attention between variable length input and output sequences
                attention_mask = torch.unsqueeze(
                    feature_mask, 2) * torch.unsqueeze(mask, -1)
                attention = generate_path(durations, attention_mask)

                # Expand sequence using predicted durations
                predicted_mean = torch.matmul(
                    attention.squeeze(1),
                    predicted_mean.transpose(1, 2)).transpose(1, 2)
                predicted_logstd = torch.matmul(
                    attention.squeeze(1),
                    predicted_logstd.transpose(1, 2)).transpose(1, 2)

        else:

            attention = None
            durations = None

        return durations, attention, predicted_mean, predicted_logstd, mask

    def latents(
        self,
        phonemes,
        pitch,
        periodicity,
        loudness,
        lengths,
        speakers,
        ratios=None,
        spectrograms=None,
        spectrogram_lengths=None):
        """Get latent representation"""
        # Scale and concatenate input features
        features, pitch, loudness = self.prepare_features(
            phonemes,
            pitch,
            periodicity,
            loudness)

        if promonet.MODEL in ['end-to-end', 'two-stage', 'vits']:

            # Encode text to text embedding and statistics for the flow model
            (
                embeddings,
                predicted_mean,
                predicted_logstd,
                feature_mask
            ) = self.prior_encoder(features, lengths)

            # Encode speaker ID
            speaker_embeddings = self.speaker_embedding(speakers).unsqueeze(-1)

            # Maybe add augmentation ratios
            if ('pitch' in promonet.INPUT_FEATURES) and promonet.AUGMENT_PITCH:
                speaker_embeddings = torch.cat(
                    (speaker_embeddings, ratios[:, None, None]),
                    dim=1)

        else:
            embeddings = None
            predicted_mean = None
            predicted_logstd = None
            feature_mask = sequence_mask(lengths)
            speaker_embeddings = None

        if self.training:

            # Mask denoting valid frames
            mask = sequence_mask(
                spectrogram_lengths,
                spectrograms.size(2)
            ).unsqueeze(1).to(spectrograms.dtype)

            if promonet.MODEL in ['hifigan', 'two-stage', 'vocoder']:

                # No duration prediction
                prior = None
                attention = None
                durations = None
                true_logstd = None

                # Replace latents
                if promonet.MODEL == 'hifigan':
                    latents = promonet.preprocess.spectrogram.linear_to_mel(
                        spectrograms)
                elif promonet.MODEL == 'two-stage':
                    latents = embeddings
                else:
                    latents = features

            else:

                # Encode linear spectrogram to latent
                latents, _, true_logstd = self.posterior_encoder(
                    spectrograms,
                    mask,
                    g=speaker_embeddings)

                # Compute corresponding prior to the latent variable
                prior = self.flow(latents, mask, g=speaker_embeddings)

                # Maybe perform duration prediction from text features
                (
                    durations,
                    attention,
                    predicted_mean,
                    predicted_logstd,
                    mask
                ) = self.duration_prediction(
                    predicted_mean,
                    predicted_logstd,
                    embeddings,
                    speaker_embeddings,
                    feature_mask,
                    mask,
                    prior
                )

            if promonet.SLICING:
                # Extract random segments for training decoder
                (
                    latents,
                    phoneme_slice,
                    pitch_slice,
                    periodicity_slice,
                    loudness_slice,
                    spectrogram_slice,
                    slice_indices
                ) = self.slice(
                    latents,
                    phonemes,
                    pitch,
                    periodicity,
                    loudness,
                    spectrograms,
                    spectrogram_lengths
                )
            else:
                # No slicing
                (
                    phoneme_slice,
                    pitch_slice,
                    periodicity_slice,
                    loudness_slice,
                    spectrogram_slice
                ) = (
                    phonemes,
                    pitch,
                    periodicity,
                    loudness,
                    spectrograms
                )
                slice_indices = None

        # Generation
        else:

            slice_indices = None
            true_logstd = None

            if promonet.MODEL in ['hifigan', 'two-stage', 'vocoder']:

                # No duration prediction
                prior = None
                attention = None
                durations = None
                true_logstd = None

                # Replace latents
                if promonet.MODEL == 'hifigan':
                    latents = promonet.preprocess.spectrogram.linear_to_mel(
                        spectrograms)
                elif promonet.MODEL == 'two-stage':
                    latents = embeddings
                else:
                    latents = features

                # Mask denoting valid frames
                mask = sequence_mask(
                    lengths,
                    lengths[0]
                ).unsqueeze(1).to(latents.dtype)

            else:

                # Mask denoting valid frames
                mask = sequence_mask(
                    lengths,
                    lengths[0]
                ).unsqueeze(1).to(embeddings.dtype)

                # Generate durations
                (
                    durations,
                    attention,
                    predicted_mean,
                    predicted_logstd,
                    mask
                ) = self.duration_prediction(
                    predicted_mean,
                    predicted_logstd,
                    embeddings,
                    speaker_embeddings,
                    feature_mask,
                    mask
                )

                # Compute the prior from text features
                prior = (
                    predicted_mean +
                    torch.randn_like(predicted_mean) *
                    torch.exp(predicted_logstd) *
                    promonet.NOISE_SCALE_INFERENCE)

                # Compute latent from the prior
                latents = self.flow(
                    prior,
                    mask,
                    g=speaker_embeddings,
                    reverse=True)

            # No slicing
            (
                phoneme_slice,
                pitch_slice,
                periodicity_slice,
                loudness_slice,
                spectrogram_slice
            ) = (
                phonemes,
                pitch,
                periodicity,
                loudness,
                spectrograms
            )

        # Maybe concat or replace latent features with acoustic features
        if promonet.MODEL != 'two-stage':
            latents = self.postprocess_latents(
                latents,
                phoneme_slice,
                pitch_slice,
                periodicity_slice,
                loudness_slice)

        return (
            latents,
            speaker_embeddings,
            mask,
            slice_indices,
            spectrogram_slice,
            durations,
            attention,
            prior,
            predicted_mean,
            predicted_logstd,
            true_logstd)

    def postprocess_latents(
        self,
        latents,
        phonemes,
        pitch,
        periodicity,
        loudness):
        """Concatenate or replace latent features with acoustic features"""
        # Maybe add pitch
        if (
            ('pitch' in promonet.INPUT_FEATURES) and
            promonet.LATENT_SHORTCUT
        ):
            if promonet.PITCH_EMBEDDING:
                pitch_embeddings = self.pitch_embedding(pitch).permute(0, 2, 1)
            else:
                pitch_embeddings = (
                    (torch.log2(pitch)[:, None] - promonet.LOG_FMIN) /
                    (promonet.LOG_FMAX - promonet.LOG_FMIN))
            latents = torch.cat((latents, pitch_embeddings), dim=1)

        # Maybe add loudness
        if (
            ('loudness' in promonet.INPUT_FEATURES) and
            promonet.LATENT_SHORTCUT
        ):
            latents = torch.cat((latents, loudness[:, None]), dim=1)

        # Maybe add periodicity
        if (
            ('periodicity' in promonet.INPUT_FEATURES) and
            promonet.LATENT_SHORTCUT
        ):
            latents = torch.cat((latents, periodicity[:, None]), dim=1)

        # Maybe add phonemes
        if promonet.LATENT_SHORTCUT:
            latents = torch.cat((latents, phonemes), dim=1)

        return latents

    def prepare_features(
        self,
        phonemes,
        pitch,
        periodicity,
        loudness):
        """Scale, concatenate, or replace input features"""
        features = phonemes if phonemes.dim() == 3 else phonemes[None]

        # Maybe add pitch features
        if 'pitch' in promonet.INPUT_FEATURES:
            if promonet.PITCH_EMBEDDING:
                pitch = promonet.convert.hz_to_bins(pitch)
                pitch_embeddings = self.pitch_embedding(pitch).permute(0, 2, 1)
            else:
                pitch_embeddings = (
                    (torch.log2(pitch)[:, None] - promonet.LOG_FMIN) /
                    (promonet.LOG_FMAX - promonet.LOG_FMIN))
            features = torch.cat((features, pitch_embeddings), dim=1)

        # Maybe add loudness features
        if 'loudness' in promonet.INPUT_FEATURES:
            normalized = promonet.loudness.normalize(loudness)
            features = torch.cat((features, normalized[:, None]), dim=1)

        # Maybe add periodicity features
        if 'periodicity' in promonet.INPUT_FEATURES:
            # TEMPORARY - if it works, make part of penn
            periodicity[loudness < promonet.SILENCE_THRESHOLD] = 0.
            features = torch.cat((features, periodicity[:, None]), dim=1)

        return features, pitch, loudness

    def slice(
        self,
        latents,
        phonemes,
        pitch,
        periodicity,
        loudness,
        spectrograms,
        spectrogram_lengths):
        """Slice features during training for latent-to-waveform generator"""
        slice_size = promonet.convert.samples_to_frames(promonet.CHUNK_SIZE)
        latents, slice_indices = random_slice_segments(
            latents,
            spectrogram_lengths,
            slice_size)

        # Maybe slice loudness
        if (
            ('loudness' in promonet.INPUT_FEATURES) and
            promonet.LATENT_SHORTCUT
        ):
            loudness_slice = promonet.model.slice_segments(
                loudness,
                slice_indices,
                slice_size)
        else:
            loudness_slice = None

        # Maybe slice periodicity
        if (
            ('periodicity' in promonet.INPUT_FEATURES) and
            promonet.LATENT_SHORTCUT
        ):
            periodicity_slice = promonet.model.slice_segments(
                periodicity,
                slice_indices,
                slice_size)
        else:
            periodicity_slice = None

        # Maybe slice pitch
        if (
            ('pitch' in promonet.INPUT_FEATURES) and
            promonet.LATENT_SHORTCUT
        ):
            pitch_slice = promonet.model.slice_segments(
                pitch,
                slice_indices,
                slice_size)
        else:
            pitch_slice = None

        # Maybe slice PPGs
        if promonet.LATENT_SHORTCUT:
            phoneme_slice = promonet.model.slice_segments(
                phonemes,
                slice_indices,
                slice_size)
        else:
            phoneme_slice = None

        # Slice spectral features
        spectrogram_slice = promonet.model.slice_segments(
            spectrograms,
            slice_indices,
            slice_size)

        return (
            latents,
            phoneme_slice,
            pitch_slice,
            periodicity_slice,
            loudness_slice,
            spectrogram_slice,
            slice_indices)


###############################################################################
# Prior encoder (e.g., phonemes -> latent)
###############################################################################


class PriorEncoder(torch.nn.Module):

  def __init__(self):
    super().__init__()

    channels = promonet.HIDDEN_CHANNELS
    if promonet.MODEL == 'vits':
        self.input_layer = torch.nn.Embedding(
            promonet.NUM_FEATURES,
            channels)
    else:
        self.input_layer = torch.nn.Conv1d(
            promonet.NUM_FEATURES,
            channels,
            promonet.KERNEL_SIZE,
            1,
            promonet.KERNEL_SIZE // 2)
    self.encoder = promonet.model.transformer.Encoder(
        channels,
        promonet.FILTER_CHANNELS)

    self.channels = (
        promonet.NUM_MELS if promonet.MODEL == 'two-stage'
        else 2 * channels)
    self.projection = torch.nn.Conv1d(
        channels,
        self.channels,
        1)
    self.scale = math.sqrt(channels)

    # Speaker conditioning
    self.cond = torch.nn.Conv1d(
        promonet.GLOBAL_CHANNELS,
        channels,
        1)

  def forward(self, features, feature_lengths):
    # Embed features
    embeddings = self.input_layer(features)
    if promonet.MODEL == 'vits':
        embeddings = embeddings.permute(0, 2, 1) * self.scale

    # Construct binary mask from lengths
    mask = torch.unsqueeze(
        sequence_mask(feature_lengths, embeddings.size(2)),
        1
    ).to(embeddings.dtype)

    # Encode masked features
    embeddings = self.encoder(embeddings * mask, mask)

    # Compute mean and variance used for constructing the prior distribution
    stats = self.projection(embeddings) * mask

    if promonet.MODEL == 'two-stage':
        embeddings = stats
        mean, logstd = None, None
    else:
        mean, logstd = torch.split(stats, self.channels // 2, dim=1)

    return embeddings, mean, logstd, mask


###############################################################################
# Posterior encoder (e.g., spectrograms -> latent)
###############################################################################


class PosteriorEncoder(torch.nn.Module):

    def __init__(self, kernel_size=5, dilation_rate=1, n_layers=16):
        super().__init__()
        self.channels = promonet.HIDDEN_CHANNELS
        self.pre = torch.nn.Conv1d(
            promonet.NUM_FFT // 2 + 1,
            self.channels,
            1)
        self.enc = promonet.model.WaveNet(
            self.channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=promonet.GLOBAL_CHANNELS)
        self.proj = torch.nn.Conv1d(
            self.channels,
            2 * self.channels,
            1)

    def forward(self, x, mask, g=None):
        x = self.pre(x) * mask
        x = self.enc(x, mask, g=g)
        stats = self.proj(x) * mask
        m, logs = torch.split(stats, self.channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * mask
        return z, m, logs


###############################################################################
# Utilities
###############################################################################


def generate_path(duration, mask):
    """Compute attention matrix from phoneme durations"""
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    pad_shape = [[0, 0], [1, 0], [0, 0]]
    pad_shape = [item for sublist in pad_shape[::-1] for item in sublist]
    path = path - torch.nn.functional.pad(path, pad_shape)[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask


def random_slice_segments(segments, lengths, segment_size):
    """Randomly slice segments along last dimension"""
    max_start_indices = lengths - segment_size + 1
    start_indices = torch.rand((len(segments),), device=segments.device)
    start_indices = (start_indices * max_start_indices).to(dtype=torch.long)
    segments = promonet.model.slice_segments(
        segments,
        start_indices,
        segment_size)
    return segments, start_indices


def sequence_mask(length, max_length=None):
    """Compute a binary mask from sequence lengths"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
