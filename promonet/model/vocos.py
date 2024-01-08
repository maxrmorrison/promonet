
import math
import functools

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
            self.backbone = vocos.models.VocosBackbone(
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
        self.head = vocos.heads.ISTFTHead(
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
