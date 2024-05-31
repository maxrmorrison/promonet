import torch

import promonet


###############################################################################
# Chunked autoregressive GAN
###############################################################################


class CARGAN(torch.nn.Module):

    def __init__(self, initial_channel, gin_channels):
        super().__init__()
        self.model = promonet.model.HiFiGAN(
            initial_channel + promonet.CARGAN_OUTPUT_SIZE,
            gin_channels)
        self.ar = Autoregressive()

        # Inference buffer
        self.buffer = torch.zeros((1, 1, promonet.CARGAN_INPUT_SIZE))

    def forward(self, x, g=None, ar=None):
        if not self.training and ar == None:
            ar = self.buffer
        ar = self.ar(ar)
        ar = ar.unsqueeze(2).repeat(1, 1, x.shape[2])
        y = self.model(torch.cat((x, ar), dim=1))
        if not self.training:
            self.buffer = y[..., -promonet.CARGAN_INPUT_SIZE:]
        return y


class Autoregressive(torch.nn.Module):

    def __init__(self):
        super().__init__()
        model = [
            torch.nn.Linear(
                promonet.CARGAN_INPUT_SIZE,
                promonet.CARGAN_HIDDEN_SIZE),
            torch.nn.LeakyReLU(.1)]
        for _ in range(3):
            model.extend([
                torch.nn.Linear(
                    promonet.CARGAN_HIDDEN_SIZE,
                    promonet.CARGAN_HIDDEN_SIZE),
                torch.nn.LeakyReLU(.1)])
        model.append(
            torch.nn.Linear(
                promonet.CARGAN_HIDDEN_SIZE,
                promonet.CARGAN_OUTPUT_SIZE))
        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x.squeeze(1))
