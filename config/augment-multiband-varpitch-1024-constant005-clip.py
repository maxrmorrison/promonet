MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-multiband-varpitch-1024-constant005-clip'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Gradients above this value are clipped to this value
GRADIENT_CLIP_GENERATOR = 10000

# Number of pitch bins
PITCH_BINS = 1024

# Type of sparsification used for ppgs
# One of ['constant', 'percentile', 'topk', None]
SPARSE_PPG_METHOD = 'constant'

# Threshold for ppg sparsification.
# In [0, 1] for 'contant' and 'percentile'; integer > 0 for 'topk'.
SPARSE_PPG_THRESHOLD = 0.05

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True
