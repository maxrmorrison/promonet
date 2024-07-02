import functools

import torch

MODULE = 'promonet'

# Configuration name
CONFIG = 'mels-ours'

# Batch size
BATCH_SIZE = 64

# Input features
INPUT_FEATURES = ['spectrogram']

# Type of sparsification used for ppgs
# One of ['constant', 'percentile', 'topk', None]
SPARSE_PPG_METHOD = None

# Only use spectral features
SPECTROGRAM_ONLY = True
