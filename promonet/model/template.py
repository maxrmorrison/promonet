import functools

import torch

import promonet


class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.input_layer = torch.nn.Conv1d(1, 8, 7, padding='same')
        self.downsample_fn = functools.partial(
            torch.nn.functional.interpolate,
            mode='linear',
            align_corners=False)
        self.blocks = torch.nn.ModuleList([
            ResidualBlock(8, 8),
            ResidualBlock(8, 16),
            ResidualBlock(16, 128),
            ResidualBlock(128, 256)])

    def forward(self, template):
        activation = torch.nn.functional.leaky_relu(
            self.input_layer(template),
            promonet.LRELU_SLOPE)
        for i in range(len(self.blocks)):
            ratio = 2 if i < 2 else 8
            activation = self.downsample_fn(
                activation,
                size=activation.shape[-1] // ratio)
            activation = self.blocks[i](activation)
            if i < len(self.blocks) - 1:
                activation = torch.nn.functional.leaky_relu(
                    activation,
                    promonet.LRELU_SLOPE)

        return activation


class ResidualBlock(torch.nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size=7):
        super().__init__()
        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')
        self.input_layer = conv_fn(input_channels, output_channels)
        self.model = torch.nn.Sequential(
            torch.nn.LeakyReLU(promonet.LRELU_SLOPE),
            conv_fn(output_channels, output_channels),
            torch.nn.LeakyReLU(promonet.LRELU_SLOPE),
            conv_fn(output_channels, output_channels))

    def forward(self, x):
        x = self.input_layer(x)
        return x + self.model(x)
