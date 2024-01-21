import functools

import torch

MODULE = 'promonet'

# Configuration name
CONFIG = 'hifigan-viterbi'

# Batch size
BATCH_SIZE = 64

# Input features
INPUT_FEATURES = ['spectrogram']

# The model to use. One of ['hifigan', 'psola', 'vits', 'vocos', 'world'].
MODEL = 'hifigan'

# Whether to use the multi-resolution spectrogram discriminator from UnivNet
MULTI_RESOLUTION_DISCRIMINATOR = True

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = False

# Only use spectral features
SPECTROGRAM_ONLY = True

# Number of training steps
STEPS = 400000
