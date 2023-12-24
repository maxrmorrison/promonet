MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-multiband-varpitch-256-multimel'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Number of items in a batch
BATCH_SIZE = 16

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Whether to use multi-mel loss
MULTI_MEL_LOSS = True

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True
