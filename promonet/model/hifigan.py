import functools
import torch

import promonet


###############################################################################
# HiFi-GAN vocoder
###############################################################################


class HiFiGAN(torch.nn.Module):

    def __init__(self, initial_channel, gin_channels):
        super().__init__()
        self.num_kernels = len(promonet.HIFIGAN_RESBLOCK_KERNEL_SIZES)
        self.num_upsamples = len(promonet.HIFIGAN_UPSAMPLE_RATES)

        # Maybe compute sampling rates of each layer
        rates = torch.tensor(promonet.HIFIGAN_UPSAMPLE_RATES).flip([0])
        rates = promonet.SAMPLE_RATE / torch.cumprod(rates, 0)
        self.sampling_rates = rates.flip([0]).to(torch.int).tolist()

        # Initial convolution
        self.conv_pre = torch.nn.Conv1d(
            initial_channel,
            promonet.HIFIGAN_UPSAMPLE_INITIAL_SIZE,
            7,
            1,
            padding=3)

        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        iterator = enumerate(zip(
            promonet.HIFIGAN_UPSAMPLE_RATES,
            promonet.HIFIGAN_UPSAMPLE_KERNEL_SIZES))
        for i, (upsample_rate, kernel_size) in iterator:
            input_channels = promonet.HIFIGAN_UPSAMPLE_INITIAL_SIZE // (2 ** i)
            output_channels = \
                promonet.HIFIGAN_UPSAMPLE_INITIAL_SIZE // (2 ** (i + 1))

            # Activations
            self.activations.append(
                torch.nn.LeakyReLU(promonet.LRELU_SLOPE))

            # Upsampling layer
            self.ups.append(torch.nn.utils.weight_norm(
                torch.nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2)))

            # Residual block
            res_iterator = zip(
                promonet.HIFIGAN_RESBLOCK_KERNEL_SIZES,
                promonet.HIFIGAN_RESBLOCK_DILATION_SIZES)
            for kernel_size, dilation_rate in res_iterator:
                self.resblocks.append(
                    Block(output_channels, kernel_size, dilation_rate))

        # Final activation
        self.activations.append(
            torch.nn.LeakyReLU(promonet.LRELU_SLOPE))

        # Final conv
        self.conv_post = torch.nn.Conv1d(
            output_channels,
            1,
            7,
            1,
            3,
            bias=False)

        # Weight initialization
        self.ups.apply(init_weights)

        # Speaker conditioning
        self.cond = torch.nn.Conv1d(
            gin_channels,
            promonet.HIFIGAN_UPSAMPLE_INITIAL_SIZE,
            1)

    def forward(self, x, g=None):
        # Initial conv
        x = self.conv_pre(x)

        # Speaker conditioning
        if g is not None:
            x = (x + self.cond(g))

        for i in range(self.num_upsamples):

            # Activation
            x = self.activations[i](x)

            # Upsampling
            x = self.ups[i](x)

            # Residual block
            for j in range(self.num_kernels):
                if j:
                    xs += self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Final activation
        x = self.activations[-1](x)

        # Final conv
        x = self.conv_post(x)

        # Bound to [-1, 1]
        return torch.tanh(x)


###############################################################################
# HiFi-GAN residual block
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


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
