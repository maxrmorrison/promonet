MODULE = 'promonet'

# Configuration name
CONFIG = 'test'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Batch size
BATCH_SIZE = 64

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# The model to use. One of ['hifigan', 'psola', 'vits', 'vocos', 'world'].
MODEL = 'vits'

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True
