import math
import torch

import promonet


###############################################################################
# VITS definition
###############################################################################


class VITS(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Input feature encoding
        self.prior_encoder = PriorEncoder()

        # Acoustic encoder
        self.posterior_encoder = PosteriorEncoder()

        # Normalizing flow
        self.flow = FlowBlock(
            promonet.VITS_CHANNELS,
            promonet.VITS_CHANNELS,
            5,
            1,
            4,
            gin_channels=promonet.GLOBAL_CHANNELS)

    def forward(self, features, spectrogram, lengths, global_features=None):
        mask = promonet.model.mask_from_lengths(
            lengths
        ).unsqueeze(1).to(features.dtype)

        # Encode text to text embedding and statistics for the flow model
        (
            embeddings,
            predicted_mean,
            predicted_logstd
        ) = self.prior_encoder(features, mask)

        if self.training:

            # Encode linear spectrogram to latent
            latents, _, true_logstd = self.posterior_encoder(
                spectrogram,
                mask,
                g=global_features)

            # Compute corresponding prior to the latent variable
            prior = self.flow(latents, mask, g=global_features)

            # Slice
            slice_size = promonet.convert.samples_to_frames(
                promonet.CHUNK_SIZE)
            latents, slice_indices = promonet.model.random_slice_segments(
                latents,
                lengths,
                slice_size)

        else:

            # Compute the prior from input features
            prior = (
                predicted_mean +
                torch.randn_like(predicted_mean) *
                torch.exp(predicted_logstd) *
                .667)

            # Compute latent from the prior
            latents = self.flow(prior, mask, g=global_features, reverse=True)
            true_logstd, slice_indices = None, None

        return (
            latents,
            (
                slice_indices,
                prior,
                predicted_mean,
                predicted_logstd,
                true_logstd
            ))


###############################################################################
# Prior encoder (i.e., inputs -> latent)
###############################################################################


class PriorEncoder(torch.nn.Module):

  def __init__(self):
    super().__init__()

    channels = promonet.VITS_CHANNELS
    self.input_layer = torch.nn.Conv1d(
        promonet.NUM_FEATURES,
        channels,
        3,
        1,
        1)
    self.encoder = Transformer(channels, promonet.VITS_PRIOR_CHANNELS)

    self.channels = 2 * channels
    self.projection = torch.nn.Conv1d(channels, self.channels, 1)
    self.scale = math.sqrt(channels)

    # Speaker conditioning
    self.cond = torch.nn.Conv1d(promonet.GLOBAL_CHANNELS, channels, 1)

  def forward(self, features, mask):
    # Embed features
    embeddings = self.input_layer(features)

    # Encode features
    embeddings = self.encoder(embeddings * mask, mask)

    # Compute mean and variance used for constructing the prior distribution
    stats = self.projection(embeddings) * mask
    mean, logstd = torch.split(stats, self.channels // 2, dim=1)

    return embeddings, mean, logstd


###############################################################################
# Posterior encoder (i.e., spectrograms -> latent)
###############################################################################


class PosteriorEncoder(torch.nn.Module):

    def __init__(self, kernel_size=5, dilation_rate=1, n_layers=16):
        super().__init__()
        self.channels = promonet.VITS_CHANNELS
        self.pre = torch.nn.Conv1d(
            promonet.NUM_FFT // 2 + 1,
            self.channels,
            1)
        self.enc = WaveNet(
            self.channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=promonet.GLOBAL_CHANNELS)
        self.proj = torch.nn.Conv1d(self.channels, 2 * self.channels, 1)

    def forward(self, x, mask, g=None):
        x = self.pre(x) * mask
        x = self.enc(x, mask, g=g)
        stats = self.proj(x) * mask
        m, logs = torch.split(stats, self.channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * mask
        return z, m, logs


###############################################################################
# Normalizing flow
###############################################################################


class FlowBlock(torch.nn.Module):

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
            self.flows.append(
                FlowLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, mask, g=g, reverse=reverse)
        return x


class FlowLayer(torch.nn.Module):

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = torch.nn.Conv1d(
            self.half_channels,
            hidden_channels,
            1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels)
        self.post = torch.nn.Conv1d(
            hidden_channels,
            self.half_channels * (2 - mean_only),
            1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels]*2, 1)
        h = self.pre(x0) * mask
        h = self.enc(h, mask, g=g)
        stats = self.post(h) * mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels]*2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet

        x1 = (x1 - m) * torch.exp(-logs) * mask
        return torch.cat([x0, x1], 1)


class Flip(torch.nn.Module):

    def forward(self, x, mask, g=None, reverse=False):
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return x


###############################################################################
# Transformer
###############################################################################


class Transformer(torch.nn.Module):

    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=.1,
        window_size=4):
        super().__init__()
        self.dropout = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, mask):
        x = x * mask
        for i in range(len(self.attn_layers)):
            y = self.attn_layers[i](
                x,
                x,
                mask.unsqueeze(2) * mask.unsqueeze(-1))
            y = self.dropout(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, mask)
            y = self.dropout(y)
            x = self.norm_layers_2[i](x + y)
        return x * mask


class MultiHeadAttention(torch.nn.Module):

    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.,
        window_size=4):
        super().__init__()
        assert channels % n_heads == 0
        # Setup layers
        self.n_heads = n_heads
        self.window_size = window_size
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.dropout = torch.nn.Dropout(p_dropout)

        # Setup relative positional embedding
        self.k_channels = channels // n_heads
        rel_stddev = self.k_channels ** -0.5
        self.emb_rel_k = torch.nn.Parameter(
            rel_stddev * torch.randn(
                1,
                window_size * 2 + 1,
                self.k_channels))
        self.emb_rel_v = torch.nn.Parameter(
            rel_stddev * torch.randn(
                1,
                window_size * 2 + 1,
                self.k_channels))

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        # Reshape (batch, channels, time) ->
        #         (batch, heads, time, channels // heads)
        batch, channels, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(
            batch,
            self.n_heads,
            self.k_channels,
            t_t).transpose(2, 3)
        key = key.view(
            batch,
            self.n_heads,
            self.k_channels,
            t_s).transpose(2, 3)
        value = value.view(
            batch,
            self.n_heads,
            self.k_channels,
            t_s).transpose(2, 3)

        # Compute attention matrix
        scores = torch.matmul(
            query / math.sqrt(self.k_channels),
            key.transpose(-2, -1))

        # Relative positional representation
        relative_embeddings = self.relative_embeddings(self.emb_rel_k, t_s)
        relative_logits = torch.matmul(
            query / math.sqrt(self.k_channels),
            relative_embeddings.unsqueeze(0).transpose(-2, -1))
        scores += self.relative_to_absolute(relative_logits)

        # Apply sequence mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        # Compute output activation
        # (batch, heads, t_t, t_s)
        attention = self.dropout(torch.nn.functional.softmax(scores, dim=-1))
        output = torch.matmul(attention, value)

        # Convert to absolute positional representation to adjust output
        output += torch.matmul(
            self.absolute_to_relative(attention),
            self.relative_embeddings(self.emb_rel_v, t_s).unsqueeze(0))

        # Reshape (batch, heads, time, channels // heads) ->
        #         (batch, channels, time]
        output = output.transpose(2, 3).contiguous().view(batch, channels, t_t)

        return output, attention

    def relative_embeddings(self, embedding, length):
        # Pad first before slice to avoid using cond ops
        pad_length = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        end = start + 2 * length - 1
        if pad_length > 0:
            padded_embedding = torch.nn.functional.pad(
                embedding,
                (0, 0, pad_length, pad_length))
        else:
            padded_embedding = embedding
        return padded_embedding[:, start:end]

    def relative_to_absolute(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()

        # Concat columns of pad to shift from relative to absolute indexing
        x = torch.nn.functional.pad(x, (0, 1))

        # Concat extra elements so to add up to shape (len + 1, 2 * len - 1)
        x_flat = x.view([batch, heads, 2 * length * length])
        x_flat = torch.nn.functional.pad(x_flat, (0, length - 1))

        # Reshape and slice out the padded elements.
        shape = (batch, heads, length + 1, 2 * length - 1)
        return x_flat.view(shape)[:, :, :length, length - 1:]

    def absolute_to_relative(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()

        # Pad along column
        x = torch.nn.functional.pad(x, (0, length - 1))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])

        # Add 0's in the beginning that will skew the elements after reshape
        x_flat = torch.nn.functional.pad(x_flat, (length, 0))
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]


class FFN(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = torch.nn.Conv1d(
            in_channels,
            filter_channels,
            kernel_size)
        self.conv_2 = torch.nn.Conv1d(
            filter_channels,
            out_channels,
            kernel_size)
        self.dropout = torch.nn.Dropout(p_dropout)

    def forward(self, x, mask):
        x = self.conv_1(self.pad(x * mask))
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv_2(self.pad(x * mask))
        return x * mask

    def pad(self, x):
        if self.kernel_size == 1:
            return x
        padding = ((self.kernel_size - 1) // 2, self.kernel_size // 2)
        return torch.nn.functional.pad(x, padding)


class LayerNorm(torch.nn.Module):

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return torch.nn.functional.layer_norm(
            x.transpose(1, -1),
            (self.channels,),
            self.gamma,
            self.beta,
            self.eps).transpose(1, -1)


###############################################################################
# WaveNet architecture
###############################################################################


class WaveNet(torch.nn.Module):

    def __init__(
            self,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=0,
            p_dropout=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = torch.nn.Dropout(p_dropout)

        # Global conditioning
        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels,
                2 * hidden_channels * n_layers,
                1)
            self.cond_layer = torch.nn.utils.weight_norm(
                cond_layer,
                name='weight')

        # WaveNet layers
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(
                hidden_channels,
                res_skip_channels,
                1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer,
                name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, mask, g=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset +
                        2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act
