import functools
import math
import torch

import promonet


###############################################################################
# Model components
###############################################################################


class CausalConv1d(torch.nn.Conv1d):
    """Causal convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True):
        causal_padding = int((kernel_size - 1) * dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=causal_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.causal_padding = causal_padding

    def forward(self, input):
        result = super().forward(input)
        if self.causal_padding != 0:
            return result[:, :, :-self.causal_padding]
        return result


class CausalTransposeConv1d(torch.nn.ConvTranspose1d):
    """Causal transpose convolution"""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
            dilation=1):

        # Omit padding
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation)

        # Number of non-causal elements to be removed
        self.causal_padding = dilation * (kernel_size - 1) + 1 - stride

    def forward(self, input):
        result = super().forward(input)
        if self.causal_padding != 0:
            result = result[:, :, :-self.causal_padding]
        return result


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


class ConvReluNorm(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        n_layers,
        p_dropout):
        super().__init__()
        self.n_layers = n_layers
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size,
            padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(torch.nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(torch.nn.Module):
    """Dialted and depthwise-separable convolution"""

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.):
        super().__init__()
        self.n_layers = n_layers
        self.drop = torch.nn.Dropout(p_dropout)
        self.convs_sep = torch.nn.ModuleList()
        self.convs_1x1 = torch.nn.ModuleList()
        self.norms_1 = torch.nn.ModuleList()
        self.norms_2 = torch.nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(torch.nn.Conv1d(
                channels,
                channels,
                kernel_size,
                groups=channels,
                dilation=dilation,
                padding=padding))
            self.convs_1x1.append(torch.nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = torch.nn.functional.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = torch.nn.functional.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


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

        if gin_channels != 0:
            cond_layer = promonet.model.CONV1D(
                gin_channels,
                2 * hidden_channels * n_layers,
                1)
            self.cond_layer = torch.nn.utils.weight_norm(
                cond_layer,
                name='weight')

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = promonet.model.CONV1D(
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

            res_skip_layer = promonet.model.CONV1D(
                hidden_channels,
                res_skip_channels,
                1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer,
                name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = promonet.model.fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


class ResBlock(torch.nn.Module):

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5)):
        super().__init__()

        # Convolutions
        conv_fn = functools.partial(
            promonet.model.causal_weight_norm_conv1d,
            channels,
            channels,
            kernel_size,
            1)
        pad_fn = functools.partial(promonet.model.get_padding, kernel_size)
        self.convs1 = torch.nn.ModuleList([
            conv_fn(pad_fn(dilation[0]), dilation[0]),
            conv_fn(pad_fn(dilation[1]), dilation[1]),
            conv_fn(pad_fn(dilation[2]), dilation[2])])
        self.convs1.apply(promonet.model.init_weights)
        self.convs2 = torch.nn.ModuleList([
            conv_fn(pad_fn()),
            conv_fn(pad_fn()),
            conv_fn(pad_fn())])
        self.convs2.apply(promonet.model.init_weights)

        # Activations
        if promonet.SNAKE:
            activation_fn = functools.partial(
                promonet.model.Snake,
                channels)
        else:
            activation_fn = functools.partial(
                torch.nn.LeakyReLU,
                negative_slope=promonet.LRELU_SLOPE)
        self.activations1 = torch.nn.ModuleList([
            activation_fn(),
            activation_fn(),
            activation_fn()])
        self.activations2 = torch.nn.ModuleList([
            activation_fn(),
            activation_fn(),
            activation_fn()])

    def forward(self, x, x_mask=None):
        iterator = zip(
            self.convs1,
            self.convs2,
            self.activations1,
            self.activations2)
        for c1, c2, a1, a2 in iterator:
            xt = a1(x)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = a2(xt)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            torch.nn.utils.remove_weight_norm(layer)


class Log(torch.nn.Module):

    def forward(self, x, x_mask, reverse=False):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        return torch.exp(x) * x_mask


class Flip(torch.nn.Module):

    def forward(self, x, mask, g=None, reverse=False):
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return x


class ElementwiseAffine(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.m = torch.nn.Parameter(torch.zeros(channels,1))
        self.logs = torch.nn.Parameter(torch.zeros(channels,1))

    def forward(self, x, x_mask, reverse=False):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1,2])
            return y, logdet
        return (x - self.m) * torch.exp(-self.logs) * x_mask


class ResidualCouplingLayer(torch.nn.Module):

    def __init__(self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = promonet.model.CONV1D(
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
        self.post = promonet.model.CONV1D(
            hidden_channels,
            self.half_channels * (2 - mean_only),
            1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels]*2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels]*2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1,2])
            return x, logdet

        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)


class ConvFlow(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        n_layers,
        num_bins=10,
        tail_bound=5.0):
        super().__init__()
        self.filter_channels = filter_channels
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = torch.nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(
            filter_channels,
            kernel_size,
            n_layers,
            p_dropout=0.)
        self.proj = torch.nn.Conv1d(
            filter_channels,
            self.half_channels * (num_bins * 3 - 1),
            1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels]*2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2) # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins:2*self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = h[..., 2 * self.num_bins:]

        x1, logabsdet = promonet.model.transform.piecewise_rational_quadratic(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails='linear',
            tail_bound=self.tail_bound)

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1,2])
        if not reverse:
            return x, logdet
        else:
            return x


class Snake(torch.nn.Module):

    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

        # Initialize alpha
        self.alpha = torch.nn.Parameter(
            torch.ones(1, channels, 1),
            requires_grad=True)

        # Maybe initialize Kaiser window
        if promonet.SNAKE_FILTER:
            self.kernel = torch.nn.Parameter(
                self.filter(),
                requires_grad=False)

    def filter(self):
        """Compute anti-aliasing low-pass filter"""
        if promonet.SNAKE_EXACT:
            return torch.tensor([
                0.0020,
                0.0094,
                -0.0255,
                -0.0577,
                0.1286,
                0.4432,
                0.4432,
                0.1286,
                -0.0577,
                -0.0255,
                0.0094,
                0.0020
            ])[None, None]

        # Create kaiser window
        length = 12
        window = torch.kaiser_window(
            length,
            periodic=False,
            beta=4.6638)

        # Compute sinc filter
        cutoff_frequency = 1 / (2. * self.scale_factor)
        aliasing_filter = 2. * cutoff_frequency * window * torch.sinc(
            2. * cutoff_frequency * math.pi *
            (torch.arange(length) - (length - 1) / 2.))

        # Normalize
        aliasing_filter /= aliasing_filter.sum()

        return aliasing_filter[None, None]

    def forward(self, x):
        # Maybe apply anti-aliasing filter
        if promonet.SNAKE_FILTER:

            # Zero-insertion interpolation
            y = torch.zeros(
                (x.shape[0], x.shape[1], self.scale_factor * x.shape[2]),
                dtype=x.dtype,
                device=x.device)
            y[..., ::2] = x * self.scale_factor

            # Treat each channel as a new batch
            shape = y.shape
            y = y.view(shape[0] * shape[1], 1, shape[2])

            # Replication padding
            padding = promonet.model.get_padding(self.kernel.shape[2])
            x = torch.nn.functional.pad(
                y,
                (padding, padding + 1),
                mode='replicate')

            # Lowpass filter
            x = torch.nn.functional.conv1d(x, self.kernel)

            # Recover channel dimension
            x = x.reshape(shape)

        # Apply snake activation
        x = x + (1. / (self.alpha + 1e-9)) * torch.sin(self.alpha * x) ** 2

        # Maybe apply anti-aliasing filter
        if promonet.SNAKE_FILTER:

            # Treat each channel as a new batch
            shape = x.shape
            x = x.view(shape[0] * shape[1], 1, shape[2])

            # Replication padding
            padding = promonet.model.get_padding(
                self.kernel.shape[2],
                1,
                self.scale_factor)
            x = torch.nn.functional.pad(
                x,
                (padding, padding),
                mode='replicate')

            # Filter and downsample
            x = torch.nn.functional.conv1d(
                x,
                self.kernel,
                stride=self.scale_factor)

            # Recover channel dimension
            x = x.reshape(shape[0], shape[1], shape[2] // self.scale_factor)

        return x