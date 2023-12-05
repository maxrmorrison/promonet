MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-slerp-varpitch-512'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Number of pitch bins
PITCH_BINS = 512

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest', 'slerp']
PPG_INTERP_METHOD = 'slerp'

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True
