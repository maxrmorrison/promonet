import numpy as np
import torch
import torchaudio

import promonet


###############################################################################
# Aggregate discriminator
###############################################################################


class Discriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        discriminators = []
        if promonet.MULTI_PERIOD_DISCRIMINATOR:
            discriminators = [DiscriminatorP(i) for i in [2, 3, 5, 7, 11]]
        if promonet.MULTI_SCALE_DISCRIMINATOR:
            discriminators.append(DiscriminatorS())
        if promonet.MULTI_RESOLUTION_DISCRIMINATOR:
            resolutions = [(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]
            discriminators.extend(
                [DiscriminatorR(i) for i in resolutions])
        if promonet.COMPLEX_MULTIBAND_DISCRIMINATOR:
            discriminators.append(DiscriminatorCMB())
        if promonet.FARGAN_DISCRIMINATOR:
            fft_sizes = [64, 128, 256, 512, 1024, 2048]
            resolutions = [[n_fft, n_fft // 4, n_fft] for n_fft in fft_sizes]
            discriminators.extend([
                DiscriminatorMagFree(resolutions[i])
                for i in range(len(resolutions))])
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

    def forward(self, x):
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

    def forward(self, audio):
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

    def forward(self, x):
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

    def forward(self, x):
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
# FARGAN discriminator
###############################################################################


class SpecDiscriminatorBase(torch.nn.Module):

    def __init__(self, layers, resolution):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.resolution = resolution
        n_fft = resolution[0]
        self.filterbank = torch.nn.Parameter(
            gen_filterbank(n_fft // 2, keep_size=True),
            requires_grad=False)
        self.init_weights()

    def forward(self, x):
        """returns array with feature maps and final score at index -1"""
        x = self.spectrogram(x).unsqueeze(1)
        output = []
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        return torch.flatten(output[-1], 1, -1), output[:-1]

    def init_weights(self):
        for m in self.modules():
            if (
                isinstance(m, torch.nn.Conv1d) or
                isinstance(m, torch.nn.ConvTranspose1d) or
                isinstance(m, torch.nn.Linear) or
                isinstance(m, torch.nn.Embedding)
            ):
                torch.nn.init.orthogonal_(m.weight.data)

    def spectrogram(self, x):
        # Get STFT window
        n_fft, hop_length, win_length = self.resolution
        window = getattr(torch, 'hann_window')(win_length).to(x.device)

        # Get magnitude stft
        x = torch.stft(
            x.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True) #[B, F, T]
        x = torch.abs(x)

        # Convert to decibel units
        return torchaudio.functional.amplitude_to_DB(
            x,
            db_multiplier=0.0,
            multiplier=20,
            amin=1e-05,
            top_db=80)


configs = {
    'stretch' : {
        64 : (0, 0),
        128: (1, 0),
        256: (2, 0),
        512: (3, 0),
        1024: (4, 0),
        2048: (5, 0)
    },
    'down' : {
        64 : (0, 0),
        128: (1, 0),
        256: (2, 0),
        512: (3, 0),
        1024: (4, 0),
        2048: (5, 0)
    }
}
class DiscriminatorMagFree(SpecDiscriminatorBase):

    def __init__(
        self,
        resolution,
        num_channels=16,
        max_channels=256,
        num_layers=5,
    ):
        stretch = configs['stretch'][resolution[0]]
        down = configs['down'][resolution[0]]

        self.num_channels = num_channels
        self.num_channels_max = max_channels
        self.num_layers = num_layers
        self.stretch = stretch
        self.down = down

        layers = []
        plan = create_3x3_conv_plan(num_layers + 1, stretch[0], down[0], stretch[1], down[1])
        in_channels = 1 + 2
        out_channels = self.num_channels
        for i in range(self.num_layers):
            layers.append(
                torch.nn.Sequential(
                    FrequencyPositionalEmbedding(),
                    torch.nn.utils.weight_norm(
                        torch.nn.Conv2d(
                            in_channels,
                            out_channels,
                            (3, 3),
                            stride=plan[i][0],
                            dilation=plan[i][1],
                            padding=plan[i][2])),
                    torch.nn.ReLU(inplace=True)))
            in_channels = out_channels + 2
            channel_factor = plan[i][0][0] * plan[i][0][1]
            out_channels = min(channel_factor * out_channels, self.num_channels_max)
        layers.append(
            torch.nn.Sequential(
                FrequencyPositionalEmbedding(),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv2d(
                        in_channels,
                        1,
                        (3, 3),
                        stride=plan[-1][0],
                        dilation=plan[-1][1],
                        padding=plan[-1][2])),
                torch.nn.Sigmoid()))

        super().__init__(layers=layers, resolution=resolution)

        # bias biases
        bias_val = 0.1
        with torch.no_grad():
            for name, weight in self.named_parameters():
                if 'bias' in name:
                    weight = weight + bias_val


class FrequencyPositionalEmbedding(torch.nn.Module):

    def forward(self, x):
        N = x.size(2)
        args = torch.arange(0, N, dtype=x.dtype, device=x.device) * torch.pi * 2 / N
        cos = torch.cos(args).reshape(1, 1, -1, 1)
        sin = torch.sin(args).reshape(1, 1, -1, 1)
        zeros = torch.zeros_like(x[:, 0:1, :, :])
        return torch.cat((x, zeros + sin, zeros + cos), dim=1)


###############################################################################
# Utilities
###############################################################################


def create_3x3_conv_plan(
    num_layers : int,
    f_stretch : int,
    f_down : int,
    t_stretch : int,
    t_down : int
):
    """ creates a stride, dilation, padding plan for a 2d conv network

    Args:
        num_layers (int): number of layers
        f_stretch (int): log_2 of stretching factor along frequency axis
        f_down (int): log_2 of downsampling factor along frequency axis
        t_stretch (int): log_2 of stretching factor along time axis
        t_down (int): log_2 of downsampling factor along time axis

    Returns:
        list(list(tuple)): list containing entries [(stride_t, stride_f), (dilation_t, dilation_f), (padding_t, padding_f)]
    """

    assert num_layers > 0 and t_stretch >= 0 and t_down >= 0 and f_stretch >= 0 and f_down >= 0
    assert f_stretch < num_layers and t_stretch < num_layers

    def process_dimension(n_layers, stretch, down):

        stack_layers = n_layers - 1

        stride_layers = min(min(down, stretch) , stack_layers)
        dilation_layers = max(min(stack_layers - stride_layers - 1, stretch - stride_layers), 0)
        final_stride = 2 ** (max(down - stride_layers, 0))

        final_dilation = 1
        if stride_layers < stack_layers and stretch - stride_layers - dilation_layers > 0:
                final_dilation = 2

        strides, dilations, paddings = [], [], []
        processed_layers = 0
        current_dilation = 1

        for _ in range(stride_layers):
            # increase receptive field and downsample via stride = 2
            strides.append(2)
            dilations.append(1)
            paddings.append(1)
            processed_layers += 1

        if processed_layers < stack_layers:
            strides.append(1)
            dilations.append(1)
            paddings.append(1)
            processed_layers += 1

        for _ in range(dilation_layers):
            # increase receptive field via dilation = 2
            strides.append(1)
            current_dilation *= 2
            dilations.append(current_dilation)
            paddings.append(current_dilation)
            processed_layers += 1

        while processed_layers < n_layers - 1:
            # fill up with std layers
            strides.append(1)
            dilations.append(current_dilation)
            paddings.append(current_dilation)
            processed_layers += 1

        # final layer
        strides.append(final_stride)
        current_dilation * final_dilation
        dilations.append(current_dilation)
        paddings.append(current_dilation)
        processed_layers += 1

        assert processed_layers == n_layers

        return strides, dilations, paddings

    t_strides, t_dilations, t_paddings = process_dimension(num_layers, t_stretch, t_down)
    f_strides, f_dilations, f_paddings = process_dimension(num_layers, f_stretch, f_down)

    plan = []

    for i in range(num_layers):
        plan.append([
            (f_strides[i], t_strides[i]),
            (f_dilations[i], t_dilations[i]),
            (f_paddings[i], t_paddings[i]),
            ])

    return plan


def gen_filterbank(N, keep_size=False):
    in_freq = (np.arange(N + 1, dtype='float32') / N * promonet.SAMPLE_RATE / 2)[None, :]
    M = N + 1 if keep_size else N
    out_freq = (np.arange(M, dtype='float32') / N * promonet.SAMPLE_RATE / 2)[:, None]
    #ERB from B.C.J Moore, An Introduction to the Psychology of Hearing, 5th Ed., page 73.
    ERB_N = 24.7 + .108 * in_freq
    delta = np.abs(in_freq - out_freq)/ERB_N
    center = (delta < .5).astype('float32')
    R = -12 * center * delta ** 2 + (1 - center) * (3 - 12 * delta)
    RE = 10. ** (R / 10.)
    norm = np.sum(RE, axis=1)
    RE = RE / norm[:, np.newaxis]
    return torch.from_numpy(RE)


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm_conv2d(*args, **kwargs)
    if not act:
        return conv
    return torch.nn.Sequential(conv, torch.nn.LeakyReLU(0.1))


def weight_norm_conv2d(*args, **kwargs):
    """Construct Conv2d layer with weight normalization"""
    return torch.nn.utils.weight_norm(torch.nn.Conv2d(*args, **kwargs))
