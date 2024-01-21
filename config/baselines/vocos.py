import functools

import torch

MODULE = 'promonet'

# Configuration name
CONFIG = 'vocos'

# Whether to use hinge loss instead of L2
ADVERSARIAL_HINGE_LOSS = True

# Batch size
BATCH_SIZE = 16

# Input features
INPUT_FEATURES = ['spectrogram']

# Whether to use the multi-resolution spectrogram discriminator from UnivNet
MULTI_RESOLUTION_DISCRIMINATOR = True

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = False

# Training optimizer
OPTIMIZER = functools.partial(
    torch.optim.AdamW,
    lr=2e-4,
    betas=(.9, .999),
    eps=1e-9)

# Only use spectral features
SPECTROGRAM_ONLY = True

# Number of training steps
STEPS = 400000

# Number of neural network layers in Vocos
VOCOS_LAYERS = 8
