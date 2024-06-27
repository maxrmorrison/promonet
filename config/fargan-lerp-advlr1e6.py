import functools

import torch

MODULE = 'promonet'

# Configuration name
CONFIG = 'fargan-lerp-advlr1e6'

# The model to use.
# One of ['fargan', 'hifigan', 'vocos', 'world'].
MODEL = 'fargan'

# Step to start using adversarial loss
ADVERSARIAL_LOSS_START_STEP = 240000

# Training batch size
BATCH_SIZE = 128

# Training sequence length
CHUNK_SIZE = 16384  # samples

# Whether to use mel spectrogram loss
MEL_LOSS = False

# Training optimizer
OPTIMIZER = functools.partial(
    torch.optim.AdamW,
    lr=2e-6,
    betas=(.9, .999),
    eps=1e-9)

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest', 'slerp']
PPG_INTERP_METHOD = 'linear'

# Whether to use multi-resolution spectral convergence loss
SPECTRAL_CONVERGENCE_LOSS = True

# Number of training steps
STEPS = 750000
