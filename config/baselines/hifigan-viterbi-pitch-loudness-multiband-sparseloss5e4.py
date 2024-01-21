import functools

import torch

MODULE = 'promonet'

# Configuration name
CONFIG = 'hifigan-viterbi-pitch-loudness-multiband-sparseloss5e4'

# Whether to use loudness augmentation
AUGMENT_LOUDNESS = True

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Batch size
BATCH_SIZE = 64

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Threshold to sparsify Mel spectrograms
DYNAMIC_RANGE_COMPRESSION_THRESHOLD = 5e-4

# Input features
INPUT_FEATURES = ['spectrogram']

# The model to use. One of ['hifigan', 'psola', 'vits', 'vocos', 'world'].
MODEL = 'hifigan'

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = False

# Whether to shift the Mels given to the Mel loss to have a minimum of zero
SPARSE_MEL_LOSS = True

# Only use spectral features
SPECTROGRAM_ONLY = True

# Number of training steps
STEPS = 400000
