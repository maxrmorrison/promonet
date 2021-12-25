import math
import torch

import promovits


###############################################################################
# Model definition
###############################################################################


# TEMPORARY - remove text-based training and generation
# class StochasticDurationPredictor(torch.nn.Module):

#     def __init__(self, in_channels, filter_channels, p_dropout=.5, n_flows=4):
#         super().__init__()
#         self.log_flow = promovits.model.modules.Log()
#         self.flows = torch.nn.ModuleList()
#         self.flows.append(promovits.model.modules.ElementwiseAffine(2))
#         for _ in range(n_flows):
#             self.flows.append(promovits.model.modules.ConvFlow(
#                 2,
#                 filter_channels,
#                 promovits.model.KERNEL_SIZE,
#                 n_layers=3))
#             self.flows.append(promovits.model.modules.Flip())

#         self.post_pre = torch.nn.Conv1d(1, filter_channels, 1)
#         self.post_proj = torch.nn.Conv1d(filter_channels, filter_channels, 1)
#         self.post_convs = promovits.model.modules.DDSConv(
#             filter_channels,
#             promovits.model.KERNEL_SIZE,
#             n_layers=3,
#             p_dropout=p_dropout)
#         self.post_flows = torch.nn.ModuleList()
#         self.post_flows.append(promovits.model.modules.ElementwiseAffine(2))
#         for _ in range(4):
#             self.post_flows.append(promovits.model.modules.ConvFlow(
#                 2,
#                 filter_channels,
#                 promovits.model.KERNEL_SIZE,
#                 n_layers=3))
#             self.post_flows.append(promovits.model.modules.Flip())

#         self.pre = torch.nn.Conv1d(in_channels, filter_channels, 1)
#         self.proj = torch.nn.Conv1d(filter_channels, filter_channels, 1)
#         self.convs = promovits.model.modules.DDSConv(
#             filter_channels,
#             promovits.model.KERNEL_SIZE,
#             n_layers=3,
#             p_dropout=p_dropout)
#         if promovits.model.GIN_CHANNELS != 0:
#             self.cond = torch.nn.Conv1d(
#                 promovits.model.GIN_CHANNELS,
#                 filter_channels,
#                 1)

#     def forward(
#         self,
#         x,
#         feature_mask,
#         w=None,
#         g=None,
#         reverse=False,
#         noise_scale=1.0):
#         x = torch.detach(x)
#         x = self.pre(x)
#         if g is not None:
#             g = torch.detach(g)
#             x = x + self.cond(g)
#         x = self.convs(x, feature_mask)
#         x = self.proj(x) * feature_mask

#         if not reverse:
#             flows = self.flows
#             assert w is not None

#             logdet_tot_q = 0
#             h_w = self.post_pre(w)
#             h_w = self.post_convs(h_w, feature_mask)
#             h_w = self.post_proj(h_w) * feature_mask
#             e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * feature_mask
#             z_q = e_q
#             for flow in self.post_flows:
#                 z_q, logdet_q = flow(z_q, feature_mask, g=(x + h_w))
#                 logdet_tot_q += logdet_q
#             z_u, z1 = torch.split(z_q, [1, 1], 1)
#             u = torch.sigmoid(z_u) * feature_mask
#             z0 = (w - u) * feature_mask
#             logdet_tot_q += torch.sum(
#                 (torch.nn.functional.logsigmoid(z_u) + torch.nn.functional.logsigmoid(-z_u)) * feature_mask,
#                 [1,2])
#             logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * feature_mask, [1, 2]) - logdet_tot_q

#             logdet_tot = 0
#             z0, logdet = self.log_flow(z0, feature_mask)
#             logdet_tot += logdet
#             z = torch.cat([z0, z1], 1)
#             for flow in flows:
#                 z, logdet = flow(z, feature_mask, g=x, reverse=reverse)
#                 logdet_tot = logdet_tot + logdet
#             nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * feature_mask, [1, 2]) - logdet_tot
#             return nll + logq # [b]
#         else:
#             flows = list(reversed(self.flows))
#             flows = flows[:-2] + [flows[-1]] # remove a useless vflow
#             z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
#             for flow in flows:
#                 z = flow(z, feature_mask, g=x, reverse=reverse)
#             z0, z1 = torch.split(z, [1, 1], 1)
#             logw = z0
#             return logw


class PhonemeEncoder(torch.nn.Module):

  def __init__(self):
    super().__init__()

    if promovits.PPG_FEATURES:
        self.input_layer = torch.nn.Conv1d(
            promovits.NUM_FEATURES,
            promovits.model.HIDDEN_CHANNELS,
            promovits.model.KERNEL_SIZE,
            1,
            promovits.model.KERNEL_SIZE // 2)
    else:
        self.input_layer = torch.nn.Embedding(
            promovits.NUM_FEATURES,
            promovits.model.HIDDEN_CHANNELS)
    self.encoder = promovits.model.attention.Encoder(
        promovits.model.HIDDEN_CHANNELS,
        promovits.model.FILTER_CHANNELS,
        promovits.model.N_HEADS,
        promovits.model.N_LAYERS,
        promovits.model.KERNEL_SIZE,
        promovits.model.P_DROPOUT)
    self.projection = torch.nn.Conv1d(
        promovits.model.HIDDEN_CHANNELS,
        2 * promovits.model.HIDDEN_CHANNELS,
        1)
    self.scale = math.sqrt(promovits.model.HIDDEN_CHANNELS)

  def forward(self, features, feature_lengths):
    # Embed features
    embeddings = self.input_layer(features)
    if not promovits.PPG_FEATURES:
        embeddings = embeddings.permute(0, 2, 1) * self.scale

    # Construct binary mask from lengths
    mask = promovits.model.sequence_mask(feature_lengths, embeddings.size(2))
    mask = torch.unsqueeze(mask, 1).to(embeddings.dtype)

    # Encode masked features
    embeddings = self.encoder(embeddings * mask, mask)

    # Compute mean and variance used for constructing the prior distribution
    stats = self.projection(embeddings) * mask
    mean, logstd = torch.split(stats, promovits.model.HIDDEN_CHANNELS, dim=1)

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
            self.flows.append(promovits.model.modules.ResidualCouplingLayer(
                channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                gin_channels=gin_channels,
                mean_only=True))
            self.flows.append(promovits.model.modules.Flip())

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
        self.pre = torch.nn.Conv1d(
            promovits.NUM_FFT // 2 + 1,
            promovits.model.HIDDEN_CHANNELS,
            1)
        self.enc = promovits.model.modules.WaveNet(
            promovits.model.HIDDEN_CHANNELS,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=promovits.model.GIN_CHANNELS)
        self.proj = torch.nn.Conv1d(
            promovits.model.HIDDEN_CHANNELS,
            2 * promovits.model.HIDDEN_CHANNELS,
            1)

    def forward(self, x, feature_lengths, g=None):
        mask = promovits.model.sequence_mask(feature_lengths, x.size(2))
        mask = torch.unsqueeze(mask, 1).to(x.dtype)
        x = self.pre(x) * mask
        x = self.enc(x, mask, g=g)
        stats = self.proj(x) * mask
        m, logs = torch.split(stats, promovits.model.HIDDEN_CHANNELS, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * mask
        return z, m, logs, mask


class LatentToAudioGenerator(torch.nn.Module):

    def __init__(self, initial_channel, gin_channels):
        super().__init__()
        self.num_kernels = len(promovits.model.RESBLOCK_KERNEL_SIZES)
        self.num_upsamples = len(promovits.model.UPSAMPLE_RATES)
        self.conv_pre = torch.nn.Conv1d(
            initial_channel,
            promovits.model.UPSAMPLE_INITIAL_SIZE,
            7,
            1,
            padding=3)

        self.ups = torch.nn.ModuleList()
        iterator = enumerate(zip(
            promovits.model.UPSAMPLE_RATES,
            promovits.model.UPSAMPLE_KERNEL_SIZES))
        for i, (upsample_rate, kernel_size) in iterator:
            self.ups.append(torch.nn.utils.weight_norm(
                torch.nn.ConvTranspose1d(
                    promovits.model.UPSAMPLE_INITIAL_SIZE // (2 ** i),
                    promovits.model.UPSAMPLE_INITIAL_SIZE // (2 ** (i + 1)),
                    kernel_size,
                    upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2)))

        self.resblocks = torch.nn.ModuleList()
        for i in range(len(self.ups)):
            channels = promovits.model.UPSAMPLE_INITIAL_SIZE // (2 ** (i + 1))
            iterator = zip(
                promovits.model.RESBLOCK_KERNEL_SIZES,
                promovits.model.RESBLOCK_DILATION_SIZES)
            for kernel_size, dilation_rate in iterator:
                self.resblocks.append(
                    promovits.model.modules.ResBlock(
                        channels,
                        kernel_size,
                        dilation_rate))

        self.conv_post = torch.nn.Conv1d(channels, 1, 7, 1, 3, bias=False)
        self.ups.apply(promovits.model.init_weights)
        self.cond = torch.nn.Conv1d(
            gin_channels,
            promovits.model.UPSAMPLE_INITIAL_SIZE,
            1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(
                x,
                promovits.model.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = torch.nn.functional.leaky_relu(x, promovits.model.LRELU_SLOPE)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for layer in self.ups:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        conv_fn = promovits.model.weight_norm_conv2d
        padding = promovits.model.get_padding(kernel_size, 1)
        self.convs = torch.nn.ModuleList([
            conv_fn(1, 32, (kernel_size, 1), (stride, 1), padding),
            conv_fn(32, 128, (kernel_size, 1), (stride, 1), padding),
            conv_fn(128, 512, (kernel_size, 1), (stride, 1), padding),
            conv_fn(512, 1024, (kernel_size, 1), (stride, 1), padding),
            conv_fn(1024, 1024, (kernel_size, 1), 1, padding)])
        self.conv_post = conv_fn(1024, 1, (3, 1), 1, (1, 0))

    def forward(self, x):
        feature_maps = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(
                x,
                promovits.model.LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        return torch.flatten(x, 1, -1), feature_maps


class DiscriminatorS(torch.nn.Module):

    def __init__(self):
        super().__init__()
        conv_fn = promovits.model.weight_norm_conv1d
        self.convs = torch.nn.ModuleList([
            conv_fn(1, 16, 15, 1, padding=7),
            conv_fn(16, 64, 41, 4, groups=4, padding=20),
            conv_fn(64, 256, 41, 4, groups=16, padding=20),
            conv_fn(256, 1024, 41, 4, groups=64, padding=20),
            conv_fn(1024, 1024, 41, 4, groups=256, padding=20),
            conv_fn(1024, 1024, 5, 1, padding=2),])
        self.conv_post = conv_fn(1024, 1, 3, 1, padding=1)

    def forward(self, x):
        feature_maps = []
        for layer in self.convs:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(
                x,
                promovits.model.LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        return torch.flatten(x, 1, -1), feature_maps


class Discriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS()] + [DiscriminatorP(i) for i in [2, 3, 5, 7, 11]])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Generator(torch.nn.Module):

    def __init__(self, n_speakers):
        super().__init__()
        self.n_speakers = n_speakers

        # Text feature encoding
        self.feature_encoder = PhonemeEncoder()
        if not promovits.PPG_FEATURES:
            # TEMPORARY - remove text-based training and generation
            raise NotImplementedError()

            # # Text-to-duration predictor
            # self.dp = StochasticDurationPredictor(
            #     promovits.model.HIDDEN_CHANNELS,
            #     192,
            #     3,
            #     0.5,
            #     4,
            #     gin_channels=promovits.GIN_CHANNELS)

        self.generator = LatentToAudioGenerator(
            promovits.model.HIDDEN_CHANNELS,
            promovits.model.GIN_CHANNELS)

        # Spectrogram encoder
        self.spectrogram_encoder = SpectrogramEncoder()

        # Normalizing flow
        self.flow = ResidualCouplingBlock(
            promovits.model.HIDDEN_CHANNELS,
            promovits.model.HIDDEN_CHANNELS,
            5,
            1,
            4,
            gin_channels=promovits.model.GIN_CHANNELS)

        # Speaker embedding
        # TODO - speaker adaptation
        self.speaker_embedding = torch.nn.Embedding(
            n_speakers,
            promovits.model.GIN_CHANNELS)

        # Pitch embedding
        if promovits.PITCH_FEATURES:
            self.pitch_embedding = torch.nn.Embedding(
                promovits.PITCH_BINS,
                promovits.PITCH_EMBEDDING_SIZE)

    def forward(
        self,
        features,
        feature_lengths,
        pitch,
        speakers,
        spectrograms=None,
        spectrogram_lengths=None,
        noise_scale=.667,
        length_scale=1,
        noise_scale_w=.8):
        """Forward pass through the network"""
        # Maybe add pitch features
        if promovits.PITCH_FEATURES:
            pitch_embeddings = self.pitch_embedding(pitch).permute(0, 2, 1)
            features = torch.cat((features, pitch_embeddings), dim=1)

        # Encode text to text embedding and statistics for the flow model
        _, predicted_mean, predicted_logstd, feature_mask = self.feature_encoder(
            features,
            feature_lengths)

        # Encode speaker ID
        # TODO - speaker adaptation
        speaker_embeddings = self.speaker_embedding(speakers).unsqueeze(-1)

        if self.training:

            # Encode linear spectrogram to latent
            latents, _, true_logstd, latent_mask = self.spectrogram_encoder(
                spectrograms,
                spectrogram_lengths,
                g=speaker_embeddings)

            # Compute corresponding prior to the latent variable
            prior = self.flow(latents, latent_mask, g=speaker_embeddings)

            if promovits.PPG_FEATURES:
                attention = None
                durations = None

            # TEMPORARY - remove text-based training and generation
            else:

                # # Compute attention mask
                # attention_mask = feature_mask * latent_mask.permute(0, 2, 1)

                # # Compute monotonic alignment
                # with torch.no_grad():
                #     s_p_sq_r = torch.exp(-2 * predicted_logstd) # [b, d, t]
                #     neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - predicted_logstd, [1], keepdim=True) # [b, 1, t_s]
                #     neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
                #     neg_cent3 = torch.matmul(z_p.transpose(1, 2), (predicted_mean * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
                #     neg_cent4 = torch.sum(-0.5 * (predicted_mean ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
                #     neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
                #     attention = promovits.model.monotonic_align.maximum_path(neg_cent, attention_mask).unsqueeze(1).detach()

                # # Calcuate duration of each feature
                # w = attention.sum(2)

                # # Predict durations from text
                # durations = self.dp(x, feature_mask, w, g=g)
                # durations = durations / torch.sum(feature_mask)

                # Expand sequence using predicted durations
                # predicted_mean = torch.matmul(
                #     attention.squeeze(1),
                #     predicted_mean.transpose(1, 2)).transpose(1, 2)
                # predicted_logstd = torch.matmul(
                #     attention.squeeze(1),
                #     predicted_logstd.transpose(1, 2)).transpose(1, 2)
                raise NotImplementedError()

            # Extract random segments of latent representation for training decoder
            latents, slice_indices = promovits.model.random_slice_segments(
                latents,
                spectrogram_lengths,
                promovits.TRAINING_CHUNK_SIZE // promovits.HOPSIZE)

        # Generation
        else:

            if promovits.PPG_FEATURES:
                latent_mask = promovits.model.sequence_mask(feature_lengths)
                latent_mask = latent_mask.unsqueeze(1).to(feature_mask.dtype)
                slice_indices = None
                attention = None
                durations = None
                true_logstd = None

            else:

                # # Predict durations from text
                # logw = self.dp(x, feature_mask, g=g, reverse=True, noise_scale=noise_scale_w)
                # w = torch.exp(logw) * feature_mask * length_scale
                # w_ceil = torch.ceil(w)

                # # Get total duration and sequence masks
                # spectrogram_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
                # latent_mask = torch.unsqueeze(promovits.model.sequence_mask(spectrogram_lengths, None), 1).to(feature_mask.dtype)

                # # Compute attention between variable length input and output sequences
                # attention_mask = torch.unsqueeze(feature_mask, 2) * torch.unsqueeze(latent_mask, -1)
                # attention = promovits.model.generate_path(w_ceil, attention_mask)

                # # Expand sequence using predicted durations
                # predicted_mean = torch.matmul(
                #     attention.squeeze(1),
                #     predicted_mean.transpose(1, 2)).transpose(1, 2)
                # predicted_logstd = torch.matmul(
                #     attention.squeeze(1),
                #     predicted_logstd.transpose(1, 2)).transpose(1, 2)

                raise NotImplementedError()

            # Compute the prior for the flow producing linear spectrograms
            prior = (
                predicted_mean +
                torch.randn_like(predicted_mean) *
                torch.exp(predicted_logstd) *
                noise_scale)

            # Compute linear spectrogram from the prior
            latents = self.flow(
                prior,
                latent_mask,
                g=speaker_embeddings,
                reverse=True)

        # Decode latent representation to waveform
        generated = self.generator(latents, g=speaker_embeddings)

        return (
            generated,
            latent_mask,
            slice_indices,
            durations,
            attention,
            prior,
            predicted_mean,
            predicted_logstd,
            true_logstd)
