import torch

import promonet


###############################################################################
# HiFi-GAN vocoder
###############################################################################


class Vocoder(torch.nn.Module):

    def __init__(self, initial_channel, gin_channels):
        super().__init__()
        self.num_kernels = len(promonet.RESBLOCK_KERNEL_SIZES)
        self.num_upsamples = len(promonet.UPSAMPLE_RATES)

        # Maybe compute sampling rates of each layer
        rates = torch.tensor(promonet.UPSAMPLE_RATES).flip([0])
        rates = promonet.SAMPLE_RATE / torch.cumprod(rates, 0)
        self.sampling_rates = rates.flip([0]).to(torch.int).tolist()

        # Initial convolution
        self.conv_pre = torch.nn.Conv1d(
            initial_channel,
            promonet.UPSAMPLE_INITIAL_SIZE,
            7,
            1,
            padding=3)

        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        iterator = enumerate(zip(
            promonet.UPSAMPLE_RATES,
            promonet.UPSAMPLE_KERNEL_SIZES))
        for i, (upsample_rate, kernel_size) in iterator:
            input_channels = promonet.UPSAMPLE_INITIAL_SIZE // (2 ** i)
            output_channels = \
                promonet.UPSAMPLE_INITIAL_SIZE // (2 ** (i + 1))

            # Activations
            self.activations.append(
                promonet.model.Snake(input_channels)
                if promonet.SNAKE else
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
                promonet.RESBLOCK_KERNEL_SIZES,
                promonet.RESBLOCK_DILATION_SIZES)
            for kernel_size, dilation_rate in res_iterator:
                self.resblocks.append(
                    promonet.model.modules.ResBlock(
                        output_channels,
                        kernel_size,
                        dilation_rate))

        # Final activation
        self.activations.append(
            promonet.model.Snake(output_channels)
            if promonet.SNAKE else
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
        self.ups.apply(promonet.model.init_weights)

        # Speaker conditioning
        self.cond = torch.nn.Conv1d(
            gin_channels,
            promonet.UPSAMPLE_INITIAL_SIZE,
            1)

    def forward(self, x, g=None):
        # Initial conv
        x = self.conv_pre(x)

        # Speaker conditioning
        if g is not None:
            x = x + self.cond(g)

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

    def remove_weight_norm(self):
        for layer in self.ups:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
