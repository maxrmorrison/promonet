import torch
from typing import Tuple

import promonet


###############################################################################
# FARGAN model definition
###############################################################################


class FARGAN(torch.nn.Module):
    """Framewise autoregressive generative adversarial network vocoder"""

    def __init__(self, num_features, global_channels):
        super().__init__()
        self.conditioning_network = ConditioningNetwork(
            num_features + global_channels)
        self.subframe_network = SubframeNetwork()

    def forward(self, features, global_features, previous_samples):
        """
        Arguments
            features
                Framewise input features
                shape=(batch, promonet.NUM_FEATURES, frames)
            global_features
                Global input features
                shape=(batch, promonet.GLOBAL_CHANNELS)
            previous_samples
                Previous waveform context
                shape=(batch, 1, promonet.NUM_PREVIOUS_SAMPLES)

        Returns
            signal
                Generated audio signal
                shape=(
                    batch,
                    1,
                    promonet.HOPSIZE * (frames - promonet.FARGAN_PREVIOUS_FRAMES))
        """
        device = features.device

        # Initialize signal
        signal = torch.zeros((features.shape[0], 0), device=device)

        # Initialize recurrent state
        states = initialize_recurrent_state(features.shape[0], device)

        # Iterate over frames
        for feature in features.permute(2, 0, 1):
            frame, previous_samples, states = self.step(
                feature,
                global_features.squeeze(2),
                previous_samples,
                states)
            signal = torch.cat([signal, frame], 1)

        return signal.unsqueeze(1)

    def remove_weight_norm(self):
        """Remove weight norm for scriptable inference"""
        self.subframe_network.remove_weight_norm()

    def step(
        self,
        features,
        global_features,
        previous_samples,
        states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Generate one frame

        Arguments
            features
                Frame features concatenated with pitch periods
                shape=(batch, promonet.NUM_FEATURES + 1)
            global_features
                Global input features
                shape=(batch, promonet.GLOBAL_CHANNELS)
            previous_samples
                Previous waveform context
                shape=(batch, 1, promonet.NUM_PREVIOUS_SAMPLES)
            states
                Recurrent model state

        Returns
            signal
                Generated frame
                shape=(batch, promonet.HOPSIZE)
            previous_samples
                Previous waveform context
                shape=(batch, 1, promonet.NUM_PREVIOUS_SAMPLES)
            states
                Recurrent model state
        """
        # Separate pitch period
        period = torch.round(features[:, -1]).to(torch.long)
        features = features[:, :-1]

        # Embed frame features
        features = self.conditioning_network(
            torch.cat((features, global_features), dim=1))

        # Initialize signal
        signal = torch.zeros((features.shape[0], 0), device=features.device)

        # Iterate over subframes
        for subframe in features.reshape(
            features.shape[0],
            2 * promonet.FARGAN_SUBFRAME_SIZE,
            promonet.FARGAN_SUBFRAMES
        ).permute(2, 0, 1):

            # Compute subframe samples
            subframe, states = self.subframe_network(
                subframe,
                previous_samples,
                period,
                states)
            signal = torch.cat([signal, subframe], dim=1)

            # Update previous samples
            previous_samples = torch.cat(
                [
                    previous_samples[:, :, promonet.FARGAN_SUBFRAME_SIZE:],
                    subframe[:, None]
                ],
                dim=2)

        return signal, previous_samples, states


###############################################################################
# FARGAN components
###############################################################################


class ConditioningNetwork(torch.nn.Sequential):
    """Input conditioning encoding

    Forward arguments
        x
            Input features
            shape=(batch, promonet.NUM_FEATURES + promonet.GLOBAL_CHANNELS)

    Forward returns
        y
            Output activation
            shape=(batch, 2 * promonet.HOPSIZE)
    """

    def __init__(self, channels):
        super().__init__(
            torch.nn.Linear(channels, channels, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(channels, channels, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(channels, 2 * promonet.HOPSIZE, bias=False),
            torch.nn.Tanh())


class SubframeNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        if promonet.FARGAN_GAIN_NORMALIZATION:
            self.input_gain_dense = torch.nn.Linear(
                2 * promonet.FARGAN_SUBFRAME_SIZE,
                1)
            self.pitch_gain_dense = torch.nn.Linear(promonet.HOPSIZE, 4)
        self.framewise_convolution = FramewiseConv()
        self.gru1 = torch.nn.GRUCell(
            promonet.HOPSIZE + 2 * promonet.FARGAN_SUBFRAME_SIZE,
            promonet.HOPSIZE,
            bias=False)
        self.gru2 = torch.nn.GRUCell(
            promonet.HOPSIZE + 2 * promonet.FARGAN_SUBFRAME_SIZE,
            promonet.HOPSIZE,
            bias=False)
        self.gru3 = torch.nn.GRUCell(
            promonet.HOPSIZE + 2 * promonet.FARGAN_SUBFRAME_SIZE,
            promonet.HOPSIZE,
            bias=False)
        self.gru1_glu = GLU(promonet.HOPSIZE)
        self.gru2_glu = GLU(promonet.HOPSIZE)
        self.gru3_glu = GLU(promonet.HOPSIZE)
        self.skip_glu = GLU(promonet.HOPSIZE)
        self.skip_dense = torch.nn.Linear(
            4 * promonet.HOPSIZE + 2 * promonet.FARGAN_SUBFRAME_SIZE,
            promonet.HOPSIZE,
            bias=False)
        self.output_layer = torch.nn.Linear(
            promonet.HOPSIZE,
            promonet.FARGAN_SUBFRAME_SIZE,
            bias=False)
        self.apply(init_weights)

    def forward(
        self,
        features,
        previous_samples,
        period,
        states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Arguments
            features
                Input subframe features
                shape=(batch, 2 * promonet.FARGAN_SUBFRAME_SIZE)
            previous_samples
                Previous waveform context
                shape=(batch, 1, promonet.NUM_PREVIOUS_SAMPLES)
            period
                Pitch period
                shape=(batch, 1)
            states
                Recurrent model state

        Returns
            signal
                Generated subframe
                shape=(batch, promonet.FARGAN_SUBFRAME_SIZE)
            states
                Recurrent model state
        """
        device = features.device

        # Make stochastic
        features_noise = additive_noise(features, self.training)

        # Extract a subframe one or two pitch periods ago
        lookback_index = previous_samples.shape[-1] - period[:, None] + torch.arange(
            promonet.FARGAN_SUBFRAME_SIZE + 4,
            device=device
        )[None] - 2
        lookback_index -= period[:, None] * (
            lookback_index >= previous_samples.shape[-1])
        pitch_lookback = torch.gather(
            previous_samples.squeeze(1),
            1,
            lookback_index)

        # Noise and gain normalization
        previous_subframe_noise = additive_noise(
            previous_samples[:, 0, -promonet.FARGAN_SUBFRAME_SIZE:],
            self.training)
        if promonet.FARGAN_GAIN_NORMALIZATION:
            gain = torch.exp(self.input_gain_dense(features_noise))
            previous_subframe_noise = previous_subframe_noise / (1e-5 + gain)
            pitch_lookback = pitch_lookback / (1e-5 + gain)
        pitch_lookback = additive_noise(pitch_lookback, self.training)

        # Embed subframe features
        subframe_input_features = torch.cat(
            (features_noise, previous_subframe_noise, pitch_lookback),
            dim=1)
        fwconv_out = additive_noise(
            self.framewise_convolution(subframe_input_features, states[3]),
            self.training)
        pitch_lookback = pitch_lookback[:, 2:-2]

        if promonet.FARGAN_GAIN_NORMALIZATION:
            pitch_gain = \
                torch.sigmoid(self.pitch_gain_dense(fwconv_out)) + 1e-5
        else:
            pitch_gain = torch.ones(1, 4, dtype=features.dtype, device=device)

        # GRU layer 1
        gru1_state = self.gru1(
            torch.cat(
                [
                    fwconv_out,
                    pitch_gain[:, 0:1] * pitch_lookback,
                    previous_subframe_noise
                ],
                dim=1),
            states[0])
        gru1_out = additive_noise(
            self.gru1_glu(additive_noise(gru1_state, self.training)),
            self.training)

        # GRU layer 2
        gru2_state = self.gru2(
            torch.cat(
                [
                    gru1_out,
                    pitch_gain[:, 1:2] * pitch_lookback,
                    previous_subframe_noise
                ],
                dim=1),
            states[1])
        gru2_out = additive_noise(
            self.gru2_glu(additive_noise(gru2_state, self.training)),
            self.training)

        # GRU layer 3
        gru3_state = self.gru3(
            torch.cat(
                [
                    gru2_out,
                    pitch_gain[:, 2:3] * pitch_lookback,
                    previous_subframe_noise
                ],
                dim=1),
            states[2])
        gru3_out = additive_noise(
            self.gru3_glu(additive_noise(gru3_state, self.training)),
            self.training)

        # Skip connection
        skip_features = torch.cat(
            [
                gru1_out,
                gru2_out,
                gru3_out,
                fwconv_out,
                pitch_gain[:, 3:4] * pitch_lookback,
                previous_subframe_noise
            ],
            dim=1)
        skip_out = self.skip_glu(
            additive_noise(
                torch.tanh(self.skip_dense(skip_features)),
                self.training))

        # Output layer
        output = torch.tanh(self.output_layer(skip_out))
        if promonet.FARGAN_GAIN_NORMALIZATION:
            output = output * gain

        # Updated recurrent state
        states = (gru1_state, gru2_state, gru3_state, subframe_input_features)

        return output, states

    def remove_weight_norm(self):
        """Remove weight norm for scriptable inference"""
        for layer in (
            self.framewise_convolution,
            self.gru1_glu,
            self.gru2_glu,
            self.gru3_glu,
            self.skip_glu
        ):
            layer.remove_weight_norm()


class FramewiseConv(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.utils.weight_norm(
                torch.nn.Linear(
                    2 * (4 * promonet.FARGAN_SUBFRAME_SIZE + 4),
                    promonet.HOPSIZE,
                    bias=False)),
            torch.nn.Tanh(),
            GLU(promonet.HOPSIZE))
        self.apply(init_weights)

    def forward(self, features, state):
        return self.model(torch.cat((features, state), -1))

    def remove_weight_norm(self):
        """Remove weight norm for scriptable inference"""
        for module in self.model:
            if isinstance(module, torch.nn.Linear):
                torch.nn.utils.remove_weight_norm(module)
            elif isinstance(module, GLU):
                module.remove_weight_norm()


class GLU(torch.nn.Module):
    """Gated linear unit"""

    def __init__(self, feat_size):
        super().__init__()
        self.gate = torch.nn.utils.weight_norm(
            torch.nn.Linear(feat_size, feat_size, bias=False))
        self.apply(init_weights)

    def forward(self, x):
        return x * torch.sigmoid(self.gate(x))

    def remove_weight_norm(self):
        torch.nn.utils.remove_weight_norm(self.gate)


###############################################################################
# Utilities
###############################################################################


def additive_noise(x, training: bool):
    """Add uniform random noise to the signal"""
    if training and promonet.FARGAN_ADDITIVE_NOISE:
        return torch.clamp(
            x + (1. / 127.) * (torch.rand_like(x) - .5),
            min=-1.,
            max=1.)
    return x


def initialize_recurrent_state(batch_size: int, device: torch.device):
    """Initialize tensors for causal inference"""
    return (
        torch.zeros(batch_size, promonet.HOPSIZE, device=device),
        torch.zeros(batch_size, promonet.HOPSIZE, device=device),
        torch.zeros(batch_size, promonet.HOPSIZE, device=device),
        torch.zeros(
            batch_size,
            4 * promonet.FARGAN_SUBFRAME_SIZE + 4,
            device=device))


def init_weights(module):
    if isinstance(module, torch.nn.GRU):
        for p in module.named_parameters():
            if p[0].startswith('weight_hh_'):
                torch.nn.init.orthogonal_(p[1])
    elif isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight.data)
