import torch

import promonet


###############################################################################
# FARGAN model definition
###############################################################################


class FARGAN(torch.nn.Module):
    """Framewise autoregressive generative adversarial network vocoder"""

    def __init__(self, num_features, global_channels):
        super().__init__()
        input_channels = num_features + global_channels
        self.input_network = torch.nn.Sequential(
            torch.nn.Linear(input_channels, input_channels),
            torch.nn.Tanh(),
            torch.nn.Linear(input_channels, promonet.HOPSIZE))
        self.subframe_network = SubframeNetwork()

    def forward(self, features, global_features):
        device = features.device

        # Initialize signal
        signal = torch.zeros((features.shape[0], 0), device=device)

        # Default to all zeros for previous samples
        previous_subframe = torch.zeros(
            (features.shape[0], promonet.FARGAN_SUBFRAME_SIZE),
            device=device)

        # Initialize recurrent state
        states = initialize_recurrent_state(features.shape[0], device)

        # Iterate over frames
        for i in range(0, features.shape[2]):
            frame, previous_subframe, states = self.step(
                features[:, :, i],
                global_features.squeeze(2),
                previous_subframe,
                states)
            signal = torch.cat([signal, frame], 1)

        return signal.unsqueeze(1)

    def step(self, features, global_features, previous_subframe, states):
        # Embed frame features
        features = self.input_network(
            torch.cat((features, global_features), dim=1))

        # Iterate over subframes
        frame = torch.zeros_like(features)
        for subframe in range(promonet.FARGAN_SUBFRAMES):

            # Compute subframe samples
            position = subframe * promonet.FARGAN_SUBFRAME_SIZE
            subframe, states = self.subframe_network(
                features[:, position:position + promonet.FARGAN_SUBFRAME_SIZE],
                previous_subframe,
                states)
            frame[:, position:position + promonet.FARGAN_SUBFRAME_SIZE] = \
                subframe
            previous_subframe = subframe

        return frame, previous_subframe, states


###############################################################################
# FARGAN components
###############################################################################


class SubframeNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        channels = promonet.FARGAN_SUBFRAME_SIZE
        self.framewise_convolution = FramewiseConv()
        self.gru1 = torch.nn.GRUCell(2 * channels, channels, bias=False)
        self.gru2 = torch.nn.GRUCell(2 * channels, channels, bias=False)
        self.gru3 = torch.nn.GRUCell(2 * channels, channels, bias=False)
        self.gru1_glu = GLU(channels)
        self.gru2_glu = GLU(channels)
        self.gru3_glu = GLU(channels)
        self.skip_glu = GLU(2 * channels)
        self.skip_dense = torch.nn.Linear(
            5 * channels,
            2 * channels,
            bias=False)
        self.output_layer = torch.nn.Linear(2 * channels, channels, bias=False)
        self.apply(init_weights)

    def forward(self, subframe, previous_subframe, states):
        subframe = additive_noise(subframe)
        previous_subframe = additive_noise(previous_subframe)

        # Embed subframe features
        fwconv_out = additive_noise(
            self.framewise_convolution(subframe, states[3]))

        # GRU layer 1
        gru1_state = self.gru1(
            torch.cat([fwconv_out, previous_subframe], 1),
            states[0])
        gru1_out = additive_noise(self.gru1_glu(additive_noise(gru1_state)))

        # GRU layer 2
        gru2_state = self.gru2(
            torch.cat([gru1_out, previous_subframe], 1),
            states[1])
        gru2_out = additive_noise(self.gru2_glu(additive_noise(gru2_state)))

        # GRU layer 3
        gru3_state = self.gru3(
            torch.cat([gru2_out, previous_subframe], 1),
            states[2])
        gru3_out = additive_noise(self.gru3_glu(additive_noise(gru3_state)))

        # Skip connection
        skip_features = torch.cat(
            [
                gru1_out,
                gru2_out,
                gru3_out,
                fwconv_out,
                previous_subframe
            ],
            1
        )
        skip_out = self.skip_glu(
            additive_noise(torch.tanh(self.skip_dense(skip_features))))

        # Output layer
        output = torch.tanh(self.output_layer(skip_out))

        return output, (gru1_state, gru2_state, gru3_state, subframe)


class FramewiseConv(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Construct model
        self.model = torch.nn.Sequential(
            torch.nn.utils.weight_norm(
                torch.nn.Linear(
                    2 * promonet.FARGAN_SUBFRAME_SIZE,
                    promonet.FARGAN_SUBFRAME_SIZE,
                    bias=False)),
            torch.nn.Tanh(),
            GLU(promonet.FARGAN_SUBFRAME_SIZE))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight.data)

    def forward(self, features, state):
        return self.model(torch.cat((features, state), -1))


class GLU(torch.nn.Module):
    """Gated linear unit"""

    def __init__(self, feat_size):
        super().__init__()
        self.gate = torch.nn.utils.weight_norm(
            torch.nn.Linear(feat_size, feat_size, bias=False))
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight.data)

    def forward(self, x):
        return x * torch.sigmoid(self.gate(x))


###############################################################################
# Utilities
###############################################################################


def additive_noise(x):
    """Add uniform random noise to the signal"""
    return torch.clamp(
        x + (1. / 127.) * (torch.rand_like(x) - .5),
        min=-1.,
        max=1.)


def initialize_recurrent_state(batch_size, device):
    """Initialize tensors for causal inference"""
    return (
        torch.zeros(batch_size, promonet.FARGAN_SUBFRAME_SIZE, device=device),
        torch.zeros(batch_size, promonet.FARGAN_SUBFRAME_SIZE, device=device),
        torch.zeros(batch_size, promonet.FARGAN_SUBFRAME_SIZE, device=device),
        torch.zeros(batch_size, promonet.FARGAN_SUBFRAME_SIZE, device=device))


def init_weights(module):
    if isinstance(module, torch.nn.GRU):
        for p in module.named_parameters():
            if p[0].startswith('weight_hh_'):
                torch.nn.init.orthogonal_(p[1])
