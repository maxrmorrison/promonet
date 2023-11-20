MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-slerp-vocos-multiband-varpitch-256-vits'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Number of items in a batch
BATCH_SIZE = 16

# Number of samples generated during training
CHUNK_SIZE = 16384

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# The model to use. One of [
#     'end-to-end',
#     'hifigan',
#     'psola',
#     'two-stage',
#     'vits',
#     'vocoder',
#     'world'
# ]
MODEL = 'end-to-end'

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest', 'slerp']
PPG_INTERP_METHOD = 'slerp'

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True

# Type of vocoder, one of ['hifigan', 'vocos']
VOCODER_TYPE = 'vocos'
