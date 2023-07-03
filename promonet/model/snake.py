import alias_free_torch
import torch


###############################################################################
# Snake activation
###############################################################################


class Snake(torch.nn.Sequential):

    def __init__(self, features, alpha=1.0):
        super().__init__(
            alias_free_torch.Activation1d(SnakeUnfiltered(features, alpha)))


###############################################################################
# Utilities
###############################################################################


class SnakeUnfiltered(torch.nn.Module):

    def __init__(self, features, alpha=1.0):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(features) * alpha)
        self.beta = torch.nn.Parameter(torch.zeros(features) * alpha)
        self.alpha.requires_grad = True
        self.beta.requires_grad = True
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return x + (
            (1.0 / (beta + self.no_div_by_zero)) *
            pow(torch.sin(x * alpha), 2))
