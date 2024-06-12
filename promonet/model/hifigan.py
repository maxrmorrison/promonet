import functools

import torch

import promonet


###############################################################################
# HiFi-GAN vocoder
###############################################################################


class HiFiGAN(torch.nn.Module):

    def __init__(self, initial_channel, gin_channels):
        super().__init__()

        # Input layer
        self.input_feature_conv = torch.nn.Conv1d(
            initial_channel,
            promonet.HIFIGAN_UPSAMPLE_INITIAL_SIZE,
            7,
            1,
            padding=3)

        # Speaker conditioning
        self.input_speaker_conv = torch.nn.Conv1d(
            gin_channels,
            promonet.HIFIGAN_UPSAMPLE_INITIAL_SIZE,
            1)

        # Rest of the model
        output_channels = (
            promonet.HIFIGAN_UPSAMPLE_INITIAL_SIZE //
            (2 ** len(promonet.HIFIGAN_UPSAMPLE_RATES)))
        self.model = torch.nn.Sequential(

            # MRF blocks
            *[
                MultiReceptiveFieldFusion(
                    promonet.HIFIGAN_UPSAMPLE_INITIAL_SIZE // (2 ** i),
                    promonet.HIFIGAN_UPSAMPLE_INITIAL_SIZE // (2 ** (i + 1)),
                    upsample_kernel_size,
                    upsample_rate
                )
                for i, (
                    upsample_kernel_size,
                    upsample_rate
                ) in enumerate(zip(
                    promonet.HIFIGAN_UPSAMPLE_KERNEL_SIZES,
                    promonet.HIFIGAN_UPSAMPLE_RATES
                ))
            ],

            # Last layer
            torch.nn.LeakyReLU(promonet.LRELU_SLOPE),
            torch.nn.Conv1d(output_channels, 1, 7, 1, 3, bias=False),

            # Output activation
            torch.nn.Tanh()
        )

    def forward(self, x, g, p):
        # Input layer
        x = self.input_feature_conv(x)

        # Speaker conditioning
        x = x + self.input_speaker_conv(g)

        return self.model(x)

    def remove_weight_norm(self):
        """Remove weight norm for scriptable inference"""
        for layer in self.model:
            if isinstance(layer, MultiReceptiveFieldFusion):
                layer.remove_weight_norm()


###############################################################################
# HiFi-GAN outermost block
###############################################################################


class MultiReceptiveFieldFusion(torch.nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        upsample_kernel_size,
        upsample_rate
    ):
        super().__init__()
        self.model = torch.nn.Sequential(

            # Input activation
            torch.nn.LeakyReLU(promonet.LRELU_SLOPE),

            # Upsampling layer
            torch.nn.utils.weight_norm(
                torch.nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    upsample_kernel_size,
                    upsample_rate,
                    padding=(upsample_kernel_size - upsample_rate) // 2)),

            # Residual block
            ResidualBlock(output_channels))

        # Weight initialization
        self.model[1].apply(init_weights)

    def forward(self, x):
        return self.model(x)

    def remove_weight_norm(self):
        """Remove weight norm for scriptable inference"""
        torch.nn.utils.remove_weight_norm(self.model[1])
        self.model[2].remove_weight_norm()


###############################################################################
# HiFi-GAN residual block
###############################################################################


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.num_kernels = len(promonet.HIFIGAN_RESBLOCK_KERNEL_SIZES)
        self.model = torch.nn.ModuleList([
            Block(channels, kernel_size, dilation_rate)
            for kernel_size, dilation_rate in zip(
                promonet.HIFIGAN_RESBLOCK_KERNEL_SIZES,
                promonet.HIFIGAN_RESBLOCK_DILATION_SIZES
            )
        ])

    def forward(self, x):
        xs = None
        for layer in self.model:
            xs = layer(x) if xs is None else xs + layer(x)
        return xs / self.num_kernels

    def remove_weight_norm(self):
        for layer in self.model:
            layer.remove_weight_norm()


###############################################################################
# HiFi-GAN inner block
###############################################################################


class Block(torch.nn.Module):

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5)):
        super().__init__()

        # Convolutions
        conv_fn = functools.partial(
            promonet.model.weight_norm_conv1d,
            channels,
            channels,
            kernel_size,
            1)
        pad_fn = functools.partial(promonet.model.get_padding, kernel_size)
        self.convs1 = torch.nn.ModuleList([
            conv_fn(pad_fn(dilation[0]), dilation[0]),
            conv_fn(pad_fn(dilation[1]), dilation[1]),
            conv_fn(pad_fn(dilation[2]), dilation[2])])
        self.convs1.apply(init_weights)
        self.convs2 = torch.nn.ModuleList([
            conv_fn(pad_fn()),
            conv_fn(pad_fn()),
            conv_fn(pad_fn())])
        self.convs2.apply(init_weights)

        # Activations
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

    def forward(self, x):
        iterator = zip(
            self.convs1,
            self.convs2,
            self.activations1,
            self.activations2)
        for c1, c2, a1, a2 in iterator:
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """Remove weight norm for scriptable inference"""
        for layer in self.convs1:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            torch.nn.utils.remove_weight_norm(layer)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
