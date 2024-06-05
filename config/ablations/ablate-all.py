import torch

MODULE = 'promonet'

# Configuration name
CONFIG = 'ablate-all'

# Whether to use loudness augmentation
AUGMENT_LOUDNESS = False

# Whether to use pitch augmentation
AUGMENT_PITCH = False

# Number of bands of A-weighted loudness
LOUDNESS_BANDS = torch.tensor([1])

# Type of sparsification used for ppgs
# One of ['constant', 'percentile', 'topk', None]
SPARSE_PPG_METHOD = None

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = False

# Whether to perform Viterbi decoding on pitch features
VITERBI_DECODE_PITCH = False
