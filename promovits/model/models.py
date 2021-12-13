import math
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm

import promovits


class StochasticDurationPredictor(nn.Module):

    def __init__(self, in_channels, filter_channels, p_dropout=.5, n_flows=4):
        super().__init__()
        self.log_flow = promovits.model.modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(promovits.model.modules.ElementwiseAffine(2))
        for _ in range(n_flows):
            self.flows.append(promovits.model.modules.ConvFlow(
                2,
                filter_channels,
                promovits.model.KERNEL_SIZE,
                n_layers=3))
            self.flows.append(promovits.model.modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = promovits.model.modules.DDSConv(
            filter_channels,
            promovits.model.KERNEL_SIZE,
            n_layers=3,
            p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(promovits.model.modules.ElementwiseAffine(2))
        for _ in range(4):
            self.post_flows.append(promovits.model.modules.ConvFlow(
                2,
                filter_channels,
                promovits.model.KERNEL_SIZE,
                n_layers=3))
            self.post_flows.append(promovits.model.modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = promovits.model.modules.DDSConv(
            filter_channels,
            promovits.model.KERNEL_SIZE,
            n_layers=3,
            p_dropout=p_dropout)
        if promovits.model.GIN_CHANNELS != 0:
            self.cond = nn.Conv1d(
                promovits.model.GIN_CHANNELS,
                filter_channels,
                1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
            logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
            return nll + logq # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]] # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class PPGEncoder(nn.Module):

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

        self.input_proj = nn.Conv1d(
            promovits.PPG_CHANNELS,
            promovits.model.HIDDEN_CHANNELS,
            promovits.model.KERNEL_SIZE,
            1,
            promovits.model.KERNEL_SIZE // 2)
        self.encoder = promovits.model.attention.Encoder(
            promovits.model.HIDDEN_CHANNELS,
            promovits.model.FILTER_CHANNELS,
            promovits.model.N_HEADS,
            promovits.model.N_LAYERS,
            promovits.model.KERNEL_SIZE,
            promovits.model.P_DROPOUT)
        self.projection = nn.Conv1d(
            promovits.model.HIDDEN_CHANNELS,
            out_channels * 2,
            1)

    def forward(self, ppgs, ppg_lengths):
        # Construct binary mask from lengths
        mask = promovits.model.sequence_mask(ppg_lengths, ppgs.size(2))
        mask = torch.unsqueeze(mask, 1).to(ppgs.dtype)

        # Embed masked ppgs
        embedded = self.encoder(self.input_proj(ppgs) * mask, mask)

        # Compute mean and variance for constructing the prior distribution
        stats = self.projection(embedded) * mask
        mean, logstd = torch.split(stats, self.out_channels, dim=1)

        return embedded, mean, logstd, mask


class TextEncoder(nn.Module):

  def __init__(self, n_vocab, out_channels):
    super().__init__()
    self.out_channels = out_channels
    self.embedding = nn.Embedding(n_vocab, promovits.model.HIDDEN_CHANNELS)
    nn.init.normal_(
        self.embedding.weight,
        0.0,
        promovits.model.HIDDEN_CHANNELS ** -0.5)
    self.encoder = promovits.model.attention.Encoder(
        promovits.model.HIDDEN_CHANNELS,
        promovits.model.FILTER_CHANNELS,
        promovits.model.N_HEADS,
        promovits.model.N_LAYERS,
        promovits.model.KERNEL_SIZE,
        promovits.model.P_DROPOUT)
    self.projection = nn.Conv1d(
        promovits.model.HIDDEN_CHANNELS,
        2 * out_channels,
        1)
    self.scale = math.sqrt(promovits.model.HIDDEN_CHANNELS)

  def forward(self, phonemes, phoneme_lengths):
    # Embed phonemes
    embeddings = self.scale * self.embedding(phonemes)  # [b, t, h]

    # Construct binary mask from lengths
    embeddings = torch.transpose(embeddings, 1, -1)  # [b, h, t]
    mask = promovits.model.sequence_mask(phoneme_lengths, embeddings.size(2))
    mask = torch.unsqueeze(mask, 1).to(embeddings.dtype)

    # Encode masked phonemes
    embeddings = self.encoder(embeddings * mask, mask)

    # Compute mean and variance used for constructing the prior distribution
    stats = self.projection(embeddings) * mask
    mean, logstd = torch.split(stats, self.out_channels, dim=1)

    return embeddings, mean, logstd, mask


class ResidualCouplingBlock(nn.Module):

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
        self.flows = nn.ModuleList()
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

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = promovits.model.modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(promovits.model.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class HiFiGANGenerator(torch.nn.Module):

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = promovits.model.modules.ResBlock1 if resblock == '1' else promovits.model.modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(promovits.model.init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, promovits.model.modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(promovits.model.get_padding(kernel_size, 1), 0))),
            weight_norm(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(promovits.model.get_padding(kernel_size, 1), 0))),
            weight_norm(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(promovits.model.get_padding(kernel_size, 1), 0))),
            weight_norm(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(promovits.model.get_padding(kernel_size, 1), 0))),
            weight_norm(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(promovits.model.get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, promovits.model.modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(1, 16, 15, 1, padding=7)),
            weight_norm(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            weight_norm(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            weight_norm(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            weight_norm(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            weight_norm(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = weight_norm(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, promovits.model.modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class Discriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        periods = [2,3,5,7,11]
        discs = [DiscriminatorS()]
        discs = discs + [DiscriminatorP(i) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Generator(nn.Module):

    def __init__(self, n_vocab, n_speakers=0, use_ppg=False):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_speakers = n_speakers
        self.use_ppg = use_ppg

        # Text feature encoding
        if use_ppg:
            self.enc_p = PPGEncoder(promovits.model.INTER_CHANNELS)
        else:
            self.enc_p = TextEncoder(n_vocab, promovits.model.INTER_CHANNELS)

            # Text-to-duration predictor
            self.dp = StochasticDurationPredictor(
                promovits.model.HIDDEN_CHANNELS,
                192,
                3,
                0.5,
                4,
                gin_channels=promovits.GIN_CHANNELS)

        self.dec = HiFiGANGenerator(
            promovits.model.INTER_CHANNELS,
            promovits.model.RESBLOCK,
            promovits.model.RESBLOCK_KERNEL_SIZES,
            promovits.model.RESBLOCK_DILATION_SIZES,
            promovits.model.UPSAMPLE_RATES,
            promovits.model.UPSAMPLE_INITIAL_CHANNEL,
            promovits.model.UPSAMPLE_KERNEL_SIZES,
            gin_channels=promovits.model.GIN_CHANNELS)

        # Spectrogram encoder
        self.enc_q = PosteriorEncoder(
            promovits.NUM_FFT // 2 + 1,
            promovits.model.INTER_CHANNELS,
            promovits.model.HIDDEN_CHANNELS,
            5,
            1,
            16,
            gin_channels=promovits.model.GIN_CHANNELS)

        # Normalizing flow
        self.flow = ResidualCouplingBlock(
            promovits.model.INTER_CHANNELS,
            promovits.model.HIDDEN_CHANNELS,
            5,
            1,
            4,
            gin_channels=promovits.model.GIN_CHANNELS)

        # Speaker embedding
        if n_speakers > 1:
            self.speaker_embedding = nn.Embedding(
                n_speakers,
                promovits.model.GIN_CHANNELS)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        """Forward pass through the network"""
        # Encode text to text embedding and statistics for the flow model
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Encode speaker ID
        if self.n_speakers > 0:
            g = self.speaker_embedding(sid).unsqueeze(-1) # [b, h, 1]
        else:
            g = None

        # Encode linear spectrogram and speaker embedding
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)

        # Encode embedded spectrogram to ground truth duration embedding
        z_p = self.flow(z, y_mask, g=g)

        # Optionally use PPGs instead of phonemes
        if self.use_ppg:
            with torch.no_grad():
                shape = (x.shape[0], x.shape[2], x.shape[2])
                attn = torch.zeros(shape, dtype=x.dtype, device=x.device)
                for i in range(x.shape[0]):
                    attn[i, :x_lengths[i], :x_lengths[i]] = torch.eye(
                        x_lengths[i],
                        dtype=x.dtype,
                        device=x.device)
                attn = attn.unsqueeze(1).detach()
                l_length = None

        else:
            # Compute attention mask
            attn_mask = x_mask * y_mask.permute(0, 2, 1)

            # Compute monotonic alignment
            with torch.no_grad():
                s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
                neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True) # [b, 1, t_s]
                neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
                neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
                neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
                neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
                attn = promovits.model.monotonic_align.maximum_path(neg_cent, attn_mask).unsqueeze(1).detach()

            # Calcuate duration of each phoneme
            w = attn.sum(2)

            # Predict durations from text
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)

        # Expand sequence using predicted durations
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # Feed random extracts of latent representations to the decoder
        z_slice, slice_indices = promovits.model.rand_slice_segments(
        z,
        y_lengths,
        promovits.TRAINING_CHUNK_SIZE // promovits.HOPSIZE)
        o = self.dec(z_slice, g=g)

        return o, l_length, attn, slice_indices, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        # Encode text to text embedding and statistics for the flow model
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Encode speaker ID
        if self.n_speakers > 0:
            g = self.speaker_embedding(sid).unsqueeze(-1) # [b, h, 1]
        else:
            g = None

        # Optionally use PPGs instead of phonemes
        if self.use_ppg:
            with torch.no_grad():
                shape = (x.shape[0], x.shape[2], x.shape[2])
                attn = torch.zeros(shape, dtype=x.dtype, device=x.device)
                for i in range(x.shape[0]):
                    attn[i, :x_lengths[i], :x_lengths[i]] = torch.eye(
                        x_lengths[i],
                        dtype=x.dtype,
                        device=x.device)
                attn = attn.detach()
            y_lengths = x_lengths
            y_mask = torch.unsqueeze(promovits.model.sequence_mask(y_lengths, None), 1)
            y_mask = y_mask.to(x_mask.dtype)

        else:
            # Predict durations from text
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)

            # Get total duration and sequence masks
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(promovits.model.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)

            # Compute attention between variable length input and output sequences
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = promovits.model.generate_path(w_ceil, attn_mask)

        # Expand sequence using predicted durations
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

        # Compute the prior for the normalizing flow producing linear spectrograms
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Compute linear spectrogram from the prior
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # Compute waveform from the linear spectrogram
        o = self.dec((z * y_mask)[:,:,:max_len], g=g)

        return o, attn, y_mask, (z, z_p, m_p, logs_p)
