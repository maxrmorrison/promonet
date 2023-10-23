MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-slerp-pitchembed128'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest', 'slerp']
PPG_INTERP_METHOD = 'slerp'

# Embedding size used to represent each pitch bin
PITCH_EMBEDDING_SIZE = 128
