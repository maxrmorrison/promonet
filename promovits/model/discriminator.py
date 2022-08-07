import functools

import torch

import promovits


###############################################################################
# Discriminator models
###############################################################################


class DiscriminatorP(torch.nn.Module):
    """Multi-period waveform discriminator"""

    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        conv_fn = promovits.model.weight_norm_conv2d
        padding = promovits.model.get_padding(kernel_size, 1)
        input_channels = promovits.NUM_FEATURES_DISCRIM
        self.convs = torch.nn.ModuleList([
            conv_fn(input_channels, 32, (kernel_size, 1),
                    (stride, 1), padding),
            conv_fn(32, 128, (kernel_size, 1), (stride, 1), padding),
            conv_fn(128, 512, (kernel_size, 1), (stride, 1), padding),
            conv_fn(512, 1024, (kernel_size, 1), (stride, 1), padding),
            conv_fn(1024, 1024, (kernel_size, 1), 1, padding)])
        self.conv_post = conv_fn(1024, 1, (3, 1), 1, (1, 0))

    def forward(self, x, pitch, periodicity, loudness, ratios=None):
        # TODO - ratios
        # Maybe add pitch conditioning
        if promovits.DISCRIM_PITCH_CONDITION:
            pitch = torch.nn.functional.interpolate(
                torch.log2(pitch)[:, None],
                scale_factor=promovits.HOPSIZE,
                mode='linear',
                align_corners=False)
            x = torch.cat((x, pitch), dim=1)

        # Maybe add periodicity conditioning
        if promovits.DISCRIM_PERIODICITY_CONDITION:
            periodicity = torch.nn.functional.interpolate(
                periodicity[:, None],
                scale_factor=promovits.HOPSIZE,
                mode='linear',
                align_corners=False)
            x = torch.cat((x, periodicity), dim=1)

        # Maybe add loudness conditioning
        if promovits.DISCRIM_LOUDNESS_CONDITION:
            loudness = promovits.loudness.normalize(loudness)
            loudness = torch.nn.functional.interpolate(
                loudness[:, None],
                scale_factor=promovits.HOPSIZE,
                mode='linear',
                align_corners=False)
            x = torch.cat((x, loudness), dim=1)

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
                promovits.model.LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        return torch.flatten(x, 1, -1), feature_maps


class DiscriminatorR(torch.nn.Module):
    """Multi-resolution spectrogram discriminator"""

    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution
        conv_fn = promovits.model.weight_norm_conv2d
        self.convs = torch.nn.ModuleList([
            conv_fn(1, 32, (3, 9), padding=(1, 4)),
            conv_fn(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            conv_fn(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            conv_fn(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
            conv_fn(32, 32, (3, 3), padding=(1, 1)),
        ])
        self.conv_post = conv_fn(32, 1, (3, 3), padding=(1, 1))

    def forward(self, audio, pitch, periodicity, loudness, ratios=None):
        # TODO - ratios
        # Compute spectral features
        features = self.spectrogram(audio)

        # Maybe add pitch conditioning
        if promovits.DISCRIM_PITCH_CONDITION:
            pitch = torch.nn.functional.interpolate(
                torch.log2(pitch)[:, None],
                size=features.shape[-1],
                mode='linear',
                align_corners=False)
            features = torch.cat((features, pitch[:, None]), dim=2)

        # Maybe add periodicity conditioning
        if promovits.DISCRIM_PERIODICITY_CONDITION:
            periodicity = torch.nn.functional.interpolate(
                periodicity[:, None],
                size=features.shape[-1],
                mode='linear',
                align_corners=False)
            features = torch.cat((features, periodicity[:, None]), dim=2)

        # Maybe add loudness conditioning
        if promovits.DISCRIM_LOUDNESS_CONDITION:
            loudness = promovits.loudness.normalize(loudness)
            loudness = torch.nn.functional.interpolate(
                loudness[:, None],
                size=features.shape[-1],
                mode='linear',
                align_corners=False)
            features = torch.cat((features, loudness[:, None]), dim=2)

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
            center=False)
        return torch.norm(x, p=2, dim=-1).unsqueeze(1)


class DiscriminatorS(torch.nn.Module):
    """Multi-scale discriminator"""

    def __init__(self):
        super().__init__()
        conv_fn = promovits.model.weight_norm_conv1d
        input_channels = promovits.NUM_FEATURES_DISCRIM
        self.convs = torch.nn.ModuleList([
            conv_fn(input_channels, 16, 15, 1, padding=7),
            conv_fn(16, 64, 41, 4, groups=4, padding=20),
            conv_fn(64, 256, 41, 4, groups=16, padding=20),
            conv_fn(256, 1024, 41, 4, groups=64, padding=20),
            conv_fn(1024, 1024, 41, 4, groups=256, padding=20),
            conv_fn(1024, 1024, 5, 1, padding=2), ])
        self.conv_post = conv_fn(1024, 1, 3, 1, padding=1)

    def forward(self, x, pitch, periodicity, loudness, ratios=None):
        # TODO - ratios
        # Maybe add pitch conditioning
        if promovits.DISCRIM_PITCH_CONDITION:
            pitch = torch.nn.functional.interpolate(
                torch.log2(pitch)[:, None],
                scale_factor=promovits.HOPSIZE,
                mode='linear',
                align_corners=False)
            x = torch.cat((x, pitch), dim=1)

        # Maybe add periodicity conditioning
        if promovits.DISCRIM_PERIODICITY_CONDITION:
            periodicity = torch.nn.functional.interpolate(
                periodicity[:, None],
                scale_factor=promovits.HOPSIZE,
                mode='linear',
                align_corners=False)
            x = torch.cat((x, periodicity), dim=1)

        # Maybe add loudness conditioning
        if promovits.DISCRIM_LOUDNESS_CONDITION:
            loudness = promovits.loudness.normalize(loudness)
            loudness = torch.nn.functional.interpolate(
                loudness[:, None],
                scale_factor=promovits.HOPSIZE,
                mode='linear',
                align_corners=False)
            x = torch.cat((x, loudness), dim=1)

        # Forward pass and save activations
        feature_maps = []
        for layer in self.convs:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(
                x,
                promovits.model.LRELU_SLOPE)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)

        return torch.flatten(x, 1, -1), feature_maps


class Discriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorP(i) for i in [2, 3, 5, 7, 11]])
        if promovits.MULTI_SCALE_DISCRIMINATOR:
            self.discriminators.append(DiscriminatorS())
        if promovits.MULTI_RESOLUTION_DISCRIMINATOR:
            resolutions = [(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]
            self.discriminators.extend(
                [DiscriminatorR(i) for i in resolutions])

    def forward(self, y, y_hat, pitch, periodicity, loudness, ratios=None):
        logits_real = []
        logits_fake = []
        feature_maps_real = []
        feature_maps_fake = []
        for discriminator in self.discriminators:
            discriminator_fn = functools.partial(
                discriminator,
                pitch=pitch,
                periodicity=periodicity,
                loudness=loudness,
                ratios=ratios)
            logit_real, feature_map_real = discriminator_fn(y)
            logit_fake, feature_map_fake = discriminator_fn(y_hat)
            logits_real.append(logit_real)
            logits_fake.append(logit_fake)
            feature_maps_real.append(feature_map_real)
            feature_maps_fake.append(feature_map_fake)

        return logits_real, logits_fake, feature_maps_real, feature_maps_fake
