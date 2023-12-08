MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-slerp-vocos-multiband-varpitch-256-varbatch-bucket4'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Number of buckets to partition training and validation data into based on length to avoid excess padding
BUCKETS = 4

# Number of samples generated during training
CHUNK_SIZE = 16384

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Maximum number of frames in a batch
MAX_TRAINING_FRAMES = 40000

# The model to use. One of [
#     'end-to-end',
#     'hifigan',
#     'psola',
#     'two-stage',
#     'vits',
#     'vocoder',
#     'world'
# ]
MODEL = 'vocoder'

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest', 'slerp']
PPG_INTERP_METHOD = 'slerp'

# Whether to slice for generator during training
SLICING = False

# Whether to use variable batch size
VARIABLE_BATCH = True

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True

# Type of vocoder, one of ['hifigan', 'vocos']
VOCODER_TYPE = 'vocos'
