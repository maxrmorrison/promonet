MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-multiband-varpitch-256-constant005'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Type of sparsification used for ppgs
# One of ['constant', 'percentile', 'topk', None]
SPARSE_PPG_METHOD = 'percentile'

# Percentage threshold for ppg sparsification (should be in [0, 1])
SPARSE_PPG_THRESHOLD = 0.85

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True
