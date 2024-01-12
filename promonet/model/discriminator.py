import torch

import promonet


###############################################################################
# Aggregate discriminator
###############################################################################


class Discriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        discriminators = [DiscriminatorP(i) for i in [2, 3, 5, 7, 11]]
        if promonet.MULTI_SCALE_DISCRIMINATOR:
            discriminators.append(DiscriminatorS())
        if promonet.MULTI_RESOLUTION_DISCRIMINATOR:
            resolutions = [(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]
            discriminators.extend(
                [DiscriminatorR(i) for i in resolutions])
        if promonet.COMPLEX_MULTIBAND_DISCRIMINATOR:
            discriminators.append(DiscriminatorCMB())
        self.discriminators = torch.nn.ModuleList(discriminators)

    def forward(self, y, y_hat, **kwargs):
        logits_real = []
        logits_fake = []
        feature_maps_real = []
        feature_maps_fake = []
        for discriminator in self.discriminators:
            logit_real, feature_map_real = discriminator(y, **kwargs)
            logit_fake, feature_map_fake = discriminator(y_hat, **kwargs)
            logits_real.append(logit_real)
            logits_fake.append(logit_fake)
            feature_maps_real.append(feature_map_real)
            feature_maps_fake.append(feature_map_fake)

        return logits_real, logits_fake, feature_maps_real, feature_maps_fake


###############################################################################
# Individual discriminators
###############################################################################


class DiscriminatorP(torch.nn.Module):
    """Multi-period waveform discriminator"""

    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        conv_fn = weight_norm_conv2d
        padding = (promonet.model.get_padding(kernel_size, 1), 0)
        input_channels = promonet.NUM_FEATURES_DISCRIM
        self.convs = torch.nn.ModuleList([
            conv_fn(input_channels, 32, (kernel_size, 1), (stride, 1), padding),
            conv_fn(32, 128, (kernel_size, 1), (stride, 1), padding),
            conv_fn(128, 512, (kernel_size, 1), (stride, 1), padding),
            conv_fn(512, 1024, (kernel_size, 1), (stride, 1), padding),
            conv_fn(1024, 1024, (kernel_size, 1), 1, padding)])
        self.conv_post = conv_fn(1024, 1, (3, 1), 1, (1, 0))

    def forward(
        self,
        x,
        pitch=None,
        periodicity=None,
        loudness=None,
        phonemes=None):
        feature_maps = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(
                x,
                promonet.LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        return torch.flatten(x, 1, -1), feature_maps


class DiscriminatorR(torch.nn.Module):
    """Multi-resolution spectrogram discriminator"""

    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution
        conv_fn = weight_norm_conv2d
        self.convs = torch.nn.ModuleList([
            conv_fn(1, 32, (3, 9), padding=(1, 4)),
            conv_fn(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            conv_fn(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            conv_fn(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            conv_fn(32, 32, (3, 3), padding=(1, 1)),
        ])
        self.conv_post = conv_fn(32, 1, (3, 3), padding=(1, 1))

    def forward(
        self,
        audio,
        pitch=None,
        periodicity=None,
        loudness=None,
        phonemes=None):
        # Compute spectral features
        features = self.spectrogram(audio)

        # Forward pass and save activations
        fmap = []
        x = features
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = torch.nn.functional.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode='reflect')
        x = torch.stft(
            x.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            return_complex=True)
        x = torch.view_as_real(x)
        return torch.norm(x, p=2, dim=-1).unsqueeze(1)


class DiscriminatorCMB(torch.nn.Module):

    def __init__(
        self,
        bands=[(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)],
    ):
        """Complex multi-band spectrogram discriminator"""
        super().__init__()
        self.window_length = promonet.WINDOW_SIZE
        self.hop_size = promonet.HOPSIZE
        self.sample_rate = promonet.SAMPLE_RATE

        n_fft = promonet.NUM_FFT // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32
        convs = lambda: torch.nn.ModuleList(
            [
                WNConv2d(1, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = torch.nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        x = torch.nn.functional.pad(
            x,
            (int((self.window_length - self.hop_size) / 2), int((self.window_length - self.hop_size) / 2)),
            mode='reflect')
        x = torch.stft(
            x.squeeze(1),
            n_fft=self.window_length,
            hop_length=self.hop_size,
            win_length=self.window_length,
            center=False,
            return_complex=True)
        x = torch.view_as_real(x)
        x = torch.norm(x, p=2, dim=-1).unsqueeze(1)
        x = torch.permute(x, (0, 1, 3, 2))

        # Split into bands
        return [x[..., b[0] : b[1]] for b in self.bands]

    def forward(
        self,
        x,
        pitch=None,
        periodicity=None,
        loudness=None,
        phonemes=None
    ):
        # Compute complex spectrogram and split into bands
        x_bands = self.spectrogram(x)

        x, fmap = [], []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)
        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(torch.nn.Module):
    """Multi-scale waveform discriminator"""

    def __init__(self):
        super().__init__()
        conv_fn = promonet.model.weight_norm_conv1d
        input_channels = promonet.NUM_FEATURES_DISCRIM
        self.convs = torch.nn.ModuleList([
            conv_fn(input_channels, 16, 15, 1, padding=7),
            conv_fn(16, 64, 41, 4, groups=4, padding=20),
            conv_fn(64, 256, 41, 4, groups=16, padding=20),
            conv_fn(256, 1024, 41, 4, groups=64, padding=20),
            conv_fn(1024, 1024, 41, 4, groups=256, padding=20),
            conv_fn(1024, 1024, 5, 1, padding=2), ])
        self.conv_post = conv_fn(1024, 1, 3, 1, padding=1)

    def forward(
        self,
        x,
        pitch=None,
        periodicity=None,
        loudness=None,
        phonemes=None):
        # Forward pass and save activations
        feature_maps = []
        for layer in self.convs:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(
                x,
                promonet.LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)

        return torch.flatten(x, 1, -1), feature_maps


###############################################################################
# Utilities
###############################################################################


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm_conv2d(*args, **kwargs)
    if not act:
        return conv
    return torch.nn.Sequential(conv, torch.nn.LeakyReLU(0.1))


def weight_norm_conv2d(*args, **kwargs):
    """Construct Conv2d layer with weight normalization"""
    return torch.nn.utils.weight_norm(torch.nn.Conv2d(*args, **kwargs))
