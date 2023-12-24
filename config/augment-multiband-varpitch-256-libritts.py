MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-multiband-varpitch-256-libritts'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Number of items in a batch
BATCH_SIZE = 16

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Dataset to use for training
TRAINING_DATASET = 'libritts'

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True
