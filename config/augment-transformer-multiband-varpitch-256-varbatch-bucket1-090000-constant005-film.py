MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-transformer-multiband-varpitch-256-varbatch-bucket1-090000-constant005-film'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Whether to use FiLM for global conditioning
FILM_CONDITIONING = False

# Maximum number of frames in a batch
MAX_TRAINING_FRAMES = 90000

# Type of sparsification used for ppgs
# One of ['constant', 'percentile', 'topk', None]
SPARSE_PPG_METHOD = 'constant'

# Threshold for ppg sparsification.
# In [0, 1] for 'contant' and 'percentile'; integer > 0 for 'topk'.
SPARSE_PPG_THRESHOLD = 0.05

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True

# Model architecture to use for vocos vocoder.
# One of ['convnext', 'transformer'].
VOCOS_ARCHITECTURE = 'transformer'
