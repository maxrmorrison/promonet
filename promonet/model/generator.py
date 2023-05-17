import math

import monotonic_align
import torch

import promonet


###############################################################################
# Model definition
###############################################################################


class StochasticDurationPredictor(torch.nn.Module):

    def __init__(self, in_channels, filter_channels, p_dropout=.5, n_flows=4):
        super().__init__()
        self.log_flow = promonet.model.modules.Log()
        self.flows = torch.nn.ModuleList()
        self.flows.append(promonet.model.modules.ElementwiseAffine(2))
        for _ in range(n_flows):
            self.flows.append(promonet.model.modules.ConvFlow(
                2,
                filter_channels,
                promonet.KERNEL_SIZE,
                n_layers=3))
            self.flows.append(promonet.model.modules.Flip())

        self.post_pre = torch.nn.Conv1d(1, filter_channels, 1)
        self.post_proj = torch.nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = promonet.model.modules.DDSConv(
            filter_channels,
            promonet.KERNEL_SIZE,
            n_layers=3,
            p_dropout=p_dropout)
        self.post_flows = torch.nn.ModuleList()
        self.post_flows.append(promonet.model.modules.ElementwiseAffine(2))
        for _ in range(4):
            self.post_flows.append(promonet.model.modules.ConvFlow(
                2,
                filter_channels,
                promonet.KERNEL_SIZE,
                n_layers=3))
            self.post_flows.append(promonet.model.modules.Flip())

        self.pre = torch.nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = torch.nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = promonet.model.modules.DDSConv(
            filter_channels,
            promonet.KERNEL_SIZE,
            n_layers=3,
            p_dropout=p_dropout)
        if promonet.GLOBAL_CHANNELS != 0:
            self.cond = torch.nn.Conv1d(
                promonet.GLOBAL_CHANNELS,
                filter_channels,
                1)

    def forward(
            self,
            x,
            feature_mask,
            w=None,
            g=None,
            reverse=False,
            noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, feature_mask)
        x = self.proj(x) * feature_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, feature_mask)
            h_w = self.post_proj(h_w) * feature_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(
                device=x.device, dtype=x.dtype) * feature_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, feature_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * feature_mask
            z0 = (w - u) * feature_mask
            logdet_tot_q += torch.sum(
                (torch.nn.functional.logsigmoid(z_u) +
                 torch.nn.functional.logsigmoid(-z_u)) * feature_mask,
                [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) +
                             (e_q ** 2)) * feature_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, feature_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, feature_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2))
                            * feature_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(
                device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, feature_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class PhonemeEncoder(torch.nn.Module):

  def __init__(self):
    super().__init__()

    channels = promonet.HIDDEN_CHANNELS
    if promonet.PPG_FEATURES or promonet.SPECTROGRAM_ONLY:
        self.input_layer = torch.nn.Conv1d(
            promonet.NUM_FEATURES,
            channels,
            promonet.KERNEL_SIZE,
            1,
            promonet.KERNEL_SIZE // 2)
    else:
        self.input_layer = torch.nn.Embedding(
            promonet.NUM_FEATURES,
            channels)
    self.encoder = promonet.model.transformer.Encoder(
        channels,
        promonet.FILTER_CHANNELS)

    self.channels = \
        promonet.NUM_FFT // 2 + 1 if promonet.TWO_STAGE else 2 * channels
    self.projection = torch.nn.Conv1d(
        channels,
        self.channels,
        1)
    self.scale = math.sqrt(self.channels)

    # Speaker conditioning
    self.cond = torch.nn.Conv1d(
        promonet.GLOBAL_CHANNELS,
        channels,
        1)

  def forward(self, features, feature_lengths):
    # Embed features
    embeddings = self.input_layer(features)
    if not promonet.PPG_FEATURES and not promonet.SPECTROGRAM_ONLY:
        embeddings = embeddings.permute(0, 2, 1) * self.scale

    # Construct binary mask from lengths
    mask = promonet.model.sequence_mask(feature_lengths, embeddings.size(2))
    mask = torch.unsqueeze(mask, 1).to(embeddings.dtype)

    # Encode masked features
    embeddings = self.encoder(embeddings * mask, mask)

    # Compute mean and variance used for constructing the prior distribution
    stats = self.projection(embeddings) * mask

    if promonet.TWO_STAGE:
        embeddings = stats
        mean, logstd = None, None
    else:
        mean, logstd = torch.split(stats, self.channels // 2, dim=1)

    return embeddings, mean, logstd, mask


class ResidualCouplingBlock(torch.nn.Module):

    def __init__(
            self,
            channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            n_flows=4,
            gin_channels=0):
        super().__init__()
        self.flows = torch.nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(promonet.model.modules.ResidualCouplingLayer(
                channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                gin_channels=gin_channels,
                mean_only=True))
            self.flows.append(promonet.model.modules.Flip())

    def forward(self, x, feature_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, feature_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, feature_mask, g=g, reverse=reverse)
        return x


class SpectrogramEncoder(torch.nn.Module):

    def __init__(self, kernel_size=5, dilation_rate=1, n_layers=16):
        super().__init__()
        self.channels = promonet.HIDDEN_CHANNELS
        self.pre = torch.nn.Conv1d(
            promonet.NUM_FFT // 2 + 1,
            self.channels,
            1)
        self.enc = promonet.model.modules.WaveNet(
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


class LatentToAudioGenerator(torch.nn.Module):

    def __init__(self, initial_channel, gin_channels):
        super().__init__()
        self.num_kernels = len(promonet.RESBLOCK_KERNEL_SIZES)
        self.num_upsamples = len(promonet.UPSAMPLE_RATES)

        # Maybe compute sampling rates of each layer
        rates = torch.tensor(promonet.UPSAMPLE_RATES).flip([0])
        rates = promonet.SAMPLE_RATE / torch.cumprod(rates, 0)
        self.sampling_rates = rates.flip([0]).to(torch.int).tolist()

        # Initial convolution
        self.conv_pre = torch.nn.Conv1d(
            initial_channel,
            promonet.UPSAMPLE_INITIAL_SIZE,
            7,
            1,
            padding=3)

        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        iterator = enumerate(zip(
            promonet.UPSAMPLE_RATES,
            promonet.UPSAMPLE_KERNEL_SIZES))
        for i, (upsample_rate, kernel_size) in iterator:
            input_channels = promonet.UPSAMPLE_INITIAL_SIZE // (2 ** i)
            output_channels = \
                promonet.UPSAMPLE_INITIAL_SIZE // (2 ** (i + 1))

            # Activations
            self.activations.append(
                promonet.model.Snake(input_channels)
                if promonet.SNAKE else
                torch.nn.LeakyReLU(promonet.LRELU_SLOPE))

            # Upsampling layer
            self.ups.append(torch.nn.utils.weight_norm(
                torch.nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2)))

            # Residual block
            res_iterator = zip(
                promonet.RESBLOCK_KERNEL_SIZES,
                promonet.RESBLOCK_DILATION_SIZES)
            for kernel_size, dilation_rate in res_iterator:
                self.resblocks.append(
                    promonet.model.modules.ResBlock(
                        output_channels,
                        kernel_size,
                        dilation_rate))

        # Final activation
        self.activations.append(
            promonet.model.Snake(output_channels)
            if promonet.SNAKE else
            torch.nn.LeakyReLU(promonet.LRELU_SLOPE))

        # Final conv
        self.conv_post = torch.nn.Conv1d(
            output_channels,
            1,
            7,
            1,
            3,
            bias=False)

        # Weight initialization
        self.ups.apply(promonet.model.init_weights)

        # Speaker conditioning
        self.cond = torch.nn.Conv1d(
            gin_channels,
            promonet.UPSAMPLE_INITIAL_SIZE,
            1)

    def forward(self, x, g=None):
        # Initial conv
        x = self.conv_pre(x)

        # Speaker conditioning
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):

            # Activation
            x = self.activations[i](x)

            # Upsampling
            x = self.ups[i](x)

            # Residual block
            for j in range(self.num_kernels):
                if j:
                    xs += self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Final activation
        x = self.activations[-1](x)

        # Final conv
        x = self.conv_post(x)

        # Bound to [-1, 1]
        return torch.tanh(x)

    def remove_weight_norm(self):
        for layer in self.ups:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class Generator(torch.nn.Module):

    def __init__(self, n_speakers=109):
        super().__init__()
        self.n_speakers = n_speakers

        # Text feature encoding
        self.feature_encoder = PhonemeEncoder()

        # Text-to-duration predictor
        if not promonet.PPG_FEATURES and not promonet.SPECTROGRAM_ONLY:
            self.dp = StochasticDurationPredictor(
                promonet.HIDDEN_CHANNELS,
                192,
                0.5,
                4)

        # Vocoder
        latent_channels = promonet.ADDITIONAL_FEATURES_LATENT
        if promonet.TWO_STAGE:
            latent_channels += promonet.NUM_FFT // 2 + 1
        elif not promonet.VOCODER:
            latent_channels += promonet.HIDDEN_CHANNELS
        self.generator = LatentToAudioGenerator(
            latent_channels,
            promonet.GLOBAL_CHANNELS)

        if not promonet.TWO_STAGE:

            # Spectrogram encoder
            self.spectrogram_encoder = SpectrogramEncoder()

            # Normalizing flow
            self.flow = ResidualCouplingBlock(
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
        if promonet.TWO_STAGE:
            self.speaker_embedding_vocoder = torch.nn.Embedding(
                n_speakers,
                promonet.SPEAKER_CHANNELS)

        # Pitch embedding
        if promonet.PITCH_FEATURES and promonet.PITCH_EMBEDDING:
            self.pitch_embedding = torch.nn.Embedding(
                promonet.PITCH_BINS,
                promonet.PITCH_EMBEDDING_SIZE)
            if promonet.LATENT_PITCH_SHORTCUT:
                self.latent_pitch_embedding = torch.nn.Embedding(
                    promonet.PITCH_BINS,
                    promonet.PITCH_EMBEDDING_SIZE)

    def forward(
        self,
        features,
        pitch,
        periodicity,
        loudness,
        lengths,
        speakers,
        ratios=None,
        spectrograms=None,
        spectrogram_lengths=None,
        noise_scale=1.,
        length_scale=1.,
        noise_scale_w=1.):
        """Generator entry point"""
        # Default augmentation ratio is 1
        if ratios is None and promonet.AUGMENT_PITCH:
            ratios = torch.ones(
                len(features),
                dtype=torch.float,
                device=features.device)

        # Get latent representation
        (
            latents,
            speaker_embeddings,
            mask,
            slice_indices,
            spectrogram_slice,
            *args
         ) = self.latents(
            features,
            pitch,
            periodicity,
            loudness,
            lengths,
            speakers,
            ratios,
            spectrograms,
            spectrogram_lengths,
            noise_scale,
            length_scale,
            noise_scale_w
        )

        # Use different speaker embedding for two-stage models
        if promonet.TWO_STAGE:
            speaker_embeddings = self.speaker_embedding_vocoder(
                speakers).unsqueeze(-1)

            # Maybe add augmentation ratios
            if promonet.PITCH_FEATURES and promonet.AUGMENT_PITCH:
                speaker_embeddings = torch.cat(
                    (speaker_embeddings, ratios[:, None, None]),
                    dim=1)

        # During first stage of two-stage generation, train the vocoder on
        # ground-truth spectrograms
        if self.training:
            if promonet.TWO_STAGE_1:
                tmp = latents.clone()
                latents = spectrogram_slice
                spectrogram_slice = tmp
            else:
                spectrogram_slice = latents

        # Decode latent representation to waveform
        generated = self.generator(latents, g=speaker_embeddings)

        return (
            generated,
            mask,
            slice_indices,
            spectrogram_slice,
            *args)

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
        spectrogram_lengths=None,
        noise_scale=1.,
        length_scale=1.,
        noise_scale_w=1.):
        """Get latent representation"""
        features = phonemes

        # Maybe add pitch features
        if promonet.PITCH_FEATURES:
            if promonet.PITCH_EMBEDDING:
                pitch = promonet.convert.hz_to_bins(pitch)
                pitch_embeddings = self.pitch_embedding(pitch).permute(0, 2, 1)
            else:
                pitch_embeddings = pitch[:, None]
            features = torch.cat((features, pitch_embeddings), dim=1)

        # Maybe add loudness features
        if promonet.LOUDNESS_FEATURES:
            loudness = promonet.loudness.normalize(loudness)
            features = torch.cat((features, loudness[:, None]), dim=1)

        # Maybe add periodicity features
        if promonet.PERIODICITY_FEATURES:
            features = torch.cat((features, periodicity[:, None]), dim=1)

        # Maybe just use the spectrogram
        if promonet.SPECTROGRAM_ONLY:
            features = spectrograms

        # Encode speaker ID
        speaker_embeddings = self.speaker_embedding(speakers).unsqueeze(-1)

        # Maybe add augmentation ratios
        if promonet.PITCH_FEATURES and promonet.AUGMENT_PITCH:
            speaker_embeddings = torch.cat(
                (speaker_embeddings, ratios[:, None, None]),
                dim=1)

        # Encode text to text embedding and statistics for the flow model
        (
            embeddings,
            predicted_mean,
            predicted_logstd,
            feature_mask
        ) = self.feature_encoder(features, lengths)

        if self.training:

            # Mask denoting valid frames
            mask = promonet.model.sequence_mask(
                spectrogram_lengths,
                spectrograms.size(2)).unsqueeze(1).to(spectrograms.dtype)

            if promonet.TWO_STAGE:
                latents = embeddings
                prior = None
                attention = None
                durations = None
                true_logstd = None

            else:
                # Encode linear spectrogram to latent
                latents, _, true_logstd = self.spectrogram_encoder(
                    spectrograms,
                    mask,
                    g=speaker_embeddings)

                # Compute corresponding prior to the latent variable
                prior = self.flow(latents, mask, g=speaker_embeddings)

                if promonet.PPG_FEATURES or promonet.SPECTROGRAM_ONLY:
                    attention = None
                    durations = None

                else:

                    # Compute attention mask
                    attention_mask = feature_mask * mask.permute(0, 2, 1)

                    # Compute monotonic alignment
                    with torch.no_grad():
                        s_p_sq_r = torch.exp(-2 * predicted_logstd)  # [b, d, t]
                        neg_cent1 = torch.sum(
                            -0.5 * math.log(2 * math.pi) - predicted_logstd,
                            [1],
                            keepdim=True)  # [b, 1, t_s]
                        # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
                        neg_cent2 = torch.matmul(
                            -0.5 * (prior ** 2).transpose(1, 2),
                            s_p_sq_r)
                        # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
                        neg_cent3 = torch.matmul(
                            prior.transpose(1, 2),
                            predicted_mean * s_p_sq_r)
                        # [b, 1, t_s]
                        neg_cent4 = torch.sum(
                            -0.5 * (predicted_mean ** 2) * s_p_sq_r,
                            [1],
                            keepdim=True)
                        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
                        attention = monotonic_align.maximum_path(
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

            # Extract random segments of latent representation for training decoder
            slice_size = promonet.convert.samples_to_frames(promonet.CHUNK_SIZE)
            latents, slice_indices = promonet.model.random_slice_segments(
                latents,
                spectrogram_lengths,
                slice_size)

            # Maybe slice loudness
            if (
                promonet.LOUDNESS_FEATURES and
                promonet.LATENT_LOUDNESS_SHORTCUT
            ):
                loudness_slice = promonet.model.slice_segments(
                    loudness,
                    slice_indices,
                    slice_size)
            else:
                loudness_slice = None

            # Maybe slice periodicity
            if (
                promonet.PERIODICITY_FEATURES and
                promonet.LATENT_PERIODICITY_SHORTCUT
            ):
                periodicity_slice = promonet.model.slice_segments(
                    periodicity,
                    slice_indices,
                    slice_size)
            else:
                periodicity_slice = None

            # Maybe slice pitch
            if (
                promonet.PITCH_FEATURES and
                promonet.LATENT_PITCH_SHORTCUT
            ):
                pitch_slice = promonet.model.slice_segments(
                    pitch,
                    slice_indices,
                    slice_size)
            else:
                pitch_slice = None

            # Maybe slice PPGs
            if promonet.LATENT_PHONEME_SHORTCUT:
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

        # Generation
        else:

            slice_indices = None
            durations = None
            true_logstd = None

            # Mask denoting valid frames
            mask = promonet.model.sequence_mask(
                lengths).unsqueeze(1).to(embeddings.dtype)

            if promonet.PPG_FEATURES or promonet.SPECTROGRAM_ONLY or promonet.TWO_STAGE:
                attention = None

            else:

                # Predict durations from text
                logw = self.dp(
                    embeddings,
                    feature_mask,
                    g=speaker_embeddings,
                    reverse=True,
                    noise_scale=noise_scale_w)
                w = torch.exp(logw) * feature_mask * length_scale
                w_ceil = torch.ceil(w)

                # Compute attention between variable length input and output sequences
                attention_mask = torch.unsqueeze(
                    feature_mask, 2) * torch.unsqueeze(mask, -1)
                attention = promonet.model.generate_path(
                    w_ceil, attention_mask)

                # Expand sequence using predicted durations
                predicted_mean = torch.matmul(
                    attention.squeeze(1),
                    predicted_mean.transpose(1, 2)).transpose(1, 2)
                predicted_logstd = torch.matmul(
                    attention.squeeze(1),
                    predicted_logstd.transpose(1, 2)).transpose(1, 2)

            if promonet.TWO_STAGE:

                latents = embeddings

            else:

                # Compute the prior from text features
                prior = (
                    predicted_mean +
                    torch.randn_like(predicted_mean) *
                    torch.exp(predicted_logstd) *
                    noise_scale)

                # Compute latent from the prior
                latents = self.flow(
                    prior,
                    mask,
                    g=speaker_embeddings,
                    reverse=True)

            # No slicing
            (
                loudness_slice,
                periodicity_slice,
                pitch_slice,
                phoneme_slice,
                spectrogram_slice
            ) = (
                loudness,
                periodicity,
                pitch,
                phonemes,
                spectrograms
            )

        # Maybe add pitch
        if (
            promonet.PITCH_FEATURES and
            promonet.LATENT_PITCH_SHORTCUT
        ):
            if promonet.PITCH_EMBEDDING:
                pitch_embeddings = self.latent_pitch_embedding(
                    pitch_slice).permute(0, 2, 1)
            else:
                pitch_embeddings = pitch_slice[:, None]
            latents = torch.cat((latents, pitch_embeddings), dim=1)

        # Maybe add loudness
        if (
            promonet.LOUDNESS_FEATURES and
            promonet.LATENT_LOUDNESS_SHORTCUT
        ):
            latents = torch.cat((latents, loudness_slice[:, None]), dim=1)

        # Maybe add periodicity
        if (
            promonet.PERIODICITY_FEATURES and
            promonet.LATENT_PERIODICITY_SHORTCUT
        ):
            latents = torch.cat((latents, periodicity_slice[:, None]), dim=1)

        # Maybe add phonemes
        if promonet.LATENT_PHONEME_SHORTCUT:
            latents = torch.cat((latents, phoneme_slice), dim=1)

        # Maybe add augmentation ratio
        if promonet.LATENT_RATIO_SHORTCUT:
            latents = torch.cat(
                (
                    latents,
                    ratios.repeat(latents.shape[2], 1, 1).permute(2, 1, 0)
                ),
                dim=1)

        # Maybe add spectrogram
        if promonet.SPECTROGRAM_ONLY:
            latents = torch.cat(
                (latents, spectrogram_slice), dim=1)

        # Maybe remove latents and keep other features
        if promonet.VOCODER:
            latents = latents[:, promonet.HIDDEN_CHANNELS:]

        # Two-stage models do not use the flow, just the feature encoder
        if promonet.TWO_STAGE:
            prior = None

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
