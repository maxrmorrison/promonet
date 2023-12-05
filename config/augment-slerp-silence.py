MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-slerp-silence'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest', 'slerp']
PPG_INTERP_METHOD = 'slerp'

# Loudness threshold (in dB) below which periodicity is set to zero
SILENCE_THRESHOLD = -60.
