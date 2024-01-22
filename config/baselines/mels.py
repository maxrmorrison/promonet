import functools

import torch

MODULE = 'promonet'

# Configuration name
CONFIG = 'mels'

# Whether to use loudness augmentation
AUGMENT_LOUDNESS = False

# Whether to use pitch augmentation
AUGMENT_PITCH = False

# Batch size
BATCH_SIZE = 64

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = False

# Input features
INPUT_FEATURES = ['spectrogram']

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = True

# Type of sparsification used for ppgs
# One of ['constant', 'percentile', 'topk', None]
SPARSE_PPG_METHOD = None

# Only use spectral features
SPECTROGRAM_ONLY = True

# Number of training steps
STEPS = 400000
