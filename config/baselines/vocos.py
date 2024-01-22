import functools

import torch

MODULE = 'promonet'

# Configuration name
CONFIG = 'vocos'

# Whether to use hinge loss instead of L2
ADVERSARIAL_HINGE_LOSS = True

# Whether to use loudness augmentation
AUGMENT_LOUDNESS = False

# Whether to use pitch augmentation
AUGMENT_PITCH = False

# Batch size
BATCH_SIZE = 16

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = False

# Input features
INPUT_FEATURES = ['spectrogram']

# The model to use. One of ['hifigan', 'psola', 'vits', 'vocos', 'world'].
MODEL = 'vocos'

# Whether to use the multi-resolution spectrogram discriminator from UnivNet
MULTI_RESOLUTION_DISCRIMINATOR = True

# Training optimizer
OPTIMIZER = functools.partial(
    torch.optim.AdamW,
    lr=2e-4,
    betas=(.9, .999),
    eps=1e-9)

# Type of sparsification used for ppgs
# One of ['constant', 'percentile', 'topk', None]
SPARSE_PPG_METHOD = None

# Only use spectral features
SPECTROGRAM_ONLY = True

# Number of training steps
STEPS = 400000

# Number of neural network layers in Vocos
VOCOS_LAYERS = 8
