import functools
import math

import alias_free_torch
import torch

import promonet


###############################################################################
# Model components
###############################################################################


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


class Flip(torch.nn.Module):

    def forward(self, x, mask, g=None, reverse=False):
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return x


class Log(torch.nn.Module):

    def forward(self, x, x_mask, reverse=False):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        return torch.exp(x) * x_mask


class ElementwiseAffine(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.m = torch.nn.Parameter(torch.zeros(channels,1))
        self.logs = torch.nn.Parameter(torch.zeros(channels,1))

    def forward(self, x, x_mask, reverse=False, g=None):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1,2])
            return y, logdet
        return (x - self.m) * torch.exp(-self.logs) * x_mask

