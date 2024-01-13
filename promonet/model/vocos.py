
import math
import functools
from typing import Optional

import torch
import vocos

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
        if promonet.VOCOS_ARCHITECTURE == 'convnext':
            self.backbone = VocosBackbone(
                promonet.VOCOS_CHANNELS,
                promonet.VOCOS_CHANNELS,
                1536,
                promonet.VOCOS_LAYERS)
        elif promonet.VOCOS_ARCHITECTURE == 'transformer':
            self.backbone = Transformer()
        else:
            raise ValueError(
                f'Vocos architecture {promonet.VOCOS_ARCHITECTURE} '
                'is not implemented')

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
            if not promonet.FILM_CONDITIONING:
                x += g

        # Infer complex STFT
        if promonet.VOCOS_ARCHITECTURE == 'convnext':
            x = self.backbone(x).transpose(1, 2)
        elif promonet.VOCOS_ARCHITECTURE == 'transformer':
            x = self.backbone(x, g)

        # Perform iSTFT to get waveform
        return self.head(x.transpose(1, 2)).unsqueeze(1)


###############################################################################
# Vocos original backbone
###############################################################################


class VocosBackbone(torch.nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = torch.nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = torch.nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = torch.nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = torch.nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x

class AdaLayerNorm(torch.nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = torch.nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale + shift
        return x


###############################################################################
# ConvNeXt block
###############################################################################


class ConvNeXtBlock(torch.nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = torch.nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = torch.nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = torch.nn.GELU()
        self.pwconv2 = torch.nn.Linear(intermediate_dim, dim)
        self.gamma = (
            torch.nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        if promonet.FILM_CONDITIONING:
            self.film = FiLM()

    def forward(self, x: torch.Tensor, z: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        if promonet.FILM_CONDITIONING:
            x = self.film(x, z)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x



###############################################################################
# Vocos ISTFT head
###############################################################################


class ISTFTHead(torch.nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio


class ISTFT(torch.nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


###############################################################################
# Transformer model
###############################################################################


class Transformer(torch.nn.Module):

    def __init__(
        self,
        num_layers=promonet.VOCOS_LAYERS,
        channels=promonet.VOCOS_CHANNELS):
        super().__init__()
        self.position = PositionalEncoding(channels)
        self.model = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(channels, 2, batch_first=True)
            for _ in range(num_layers)])
        if promonet.FILM_CONDITIONING:
            self.film = torch.nn.ModuleList([FiLM() for _ in range(num_layers)])

    def forward(self, x, z):
        x = self.position(x.permute(2, 0, 1)).permute(1, 2, 0)
        for layer, film in zip(self.model, self.film):
            x = layer(x)
            if promonet.FILM_CONDITIONING:
                x = film(x, z)
        return x


###############################################################################
# Utilities
###############################################################################


class FiLM(torch.nn.Module):

    def __init__(self):
        self.gamma = torch.nn.Conv1d(
            promonet.VOCOS_CHANNELS,
            promonet.VOCOS_CHANNELS,
            1)
        self.beta = torch.nn.Conv1d(
            promonet.VOCOS_CHANNELS,
            promonet.VOCOS_CHANNELS,
            1)

    def forward(self, x, z):
        return self.gamma(z) * x + self.beta(z)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, channels, dropout=.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        index = torch.arange(max_len).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, channels, 2) * (-math.log(10000.0) / channels))
        encoding = torch.zeros(max_len, 1, channels)
        encoding[:, 0, 0::2] = torch.sin(index * frequency)
        encoding[:, 0, 1::2] = torch.cos(index * frequency)
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        if x.size(0) > self.encoding.size(0):
            raise ValueError('size is too large')
        return self.dropout(x + self.encoding[:x.size(0)])


def mask_from_lengths(lengths, padding=0):
    """Create boolean mask from sequence lengths and offset to start"""
    x = torch.arange(
        lengths.max() + 2 * padding,
        dtype=lengths.dtype,
        device=lengths.device)
    return x.unsqueeze(0) - 2 * padding < lengths.unsqueeze(1)
