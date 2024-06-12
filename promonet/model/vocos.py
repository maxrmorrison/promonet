import torch

import promonet


###############################################################################
# Vocos vocoder
###############################################################################


class Vocos(torch.nn.Module):

    def __init__(self, initial_channel, gin_channels):
        super().__init__()

        # Input feature projection
        self.conv_pre = torch.nn.Conv1d(
            initial_channel,
            promonet.VOCOS_CHANNELS,
            7,
            1,
            padding='same')

        # Model architecture
        self.backbone = VocosBackbone(
            promonet.VOCOS_CHANNELS,
            promonet.VOCOS_CHANNELS,
            promonet.VOCOS_LAYERS)

        # Differentiable iSTFT
        self.head = ISTFTHead(
            promonet.VOCOS_CHANNELS,
            promonet.NUM_FFT,
            promonet.HOPSIZE)

        # Speaker conditioning
        self.cond = torch.nn.Conv1d(
            gin_channels,
            promonet.VOCOS_CHANNELS,
            1)

    def forward(self, x, g=None):
        # Initial conv
        x = self.conv_pre(x)

        # Speaker conditioning
        if g is not None:
            g = self.cond(g)
            x += g

        # Infer complex STFT
        x = self.backbone(x, g)

        # Perform iSTFT to get waveform
        return self.head(x)


###############################################################################
# Vocos architecture
###############################################################################


class VocosBackbone(torch.nn.Module):

    def __init__(
        self,
        input_channels: int,
        dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = torch.nn.Conv1d(
            input_channels,
            dim,
            kernel_size=7,
            padding=3)
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6)
        self.convnext = torch.nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    layer_scale_init_value=1 / num_layers)
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = torch.nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, g):
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, g)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x.transpose(1, 2)


###############################################################################
# ConvNeXt block
###############################################################################


class ConvNeXtBlock(torch.nn.Module):

    def __init__(self, dim, layer_scale_init_value):
        super().__init__()
        self.dwconv = torch.nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim)
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(dim, promonet.VOCOS_POINTWISE_CHANNELS)
        self.act = torch.nn.GELU()
        self.pwconv2 = torch.nn.Linear(promonet.VOCOS_POINTWISE_CHANNELS, dim)
        self.gamma = (
            torch.nn.Parameter(
                layer_scale_init_value * torch.ones(dim),
                requires_grad=True))

    def forward(self, x, g):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


###############################################################################
# Vocos ISTFT head
###############################################################################


class ISTFTHead(torch.nn.Module):

    def __init__(self, dim, n_fft, hop_length):
        super().__init__()
        self.out = torch.nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft)

    def forward(self, x):
        x = self.out(x.transpose(1, 2)).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)
        return self.istft(S).unsqueeze(1)


class ISTFT(torch.nn.Module):

    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer('window', window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        B, N, T = spec.shape
        pad = (self.win_length - self.hop_length) // 2

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm='backward')
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        return y / window_envelope
