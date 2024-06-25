MODULE = 'promonet'

# Configuration name
CONFIG = 'promonet-lerp-zeroshot'

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest', 'slerp']
PPG_INTERP_METHOD = 'linear'

# Number of training steps
STEPS = 1000000

# Whether to use WavLM x-vectors for zero-shot speaker conditioning
ZERO_SHOT = True
