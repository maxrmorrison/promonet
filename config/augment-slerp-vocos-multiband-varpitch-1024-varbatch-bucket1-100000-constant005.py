MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-slerp-vocos-multiband-varpitch-1024-varbatch-bucket1-100000-constant005'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Number of buckets to partition training and validation data into based on length to avoid excess padding
BUCKETS = 1

# Number of samples generated during training
CHUNK_SIZE = 16384

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Maximum number of frames in a batch
MAX_TRAINING_FRAMES = 100000

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

# Number of pitch bins
PITCH_BINS = 1024

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest', 'slerp']
PPG_INTERP_METHOD = 'slerp'

# Whether to slice for generator during training
SLICING = False

# Whether to use sparse ppgs
SPARSE_PPGS = True

# Type of sparsification used for ppgs
# Available methods are ['constant_threshold', 'percent_threshold', 'top_n']
SPARSE_METHOD = 'constant_threshold'

# Constant threshold for ppg sparsification (should be in [0, 1])
SPARSE_THRESHOLD = 0.05

# Whether to use variable batch size
VARIABLE_BATCH = True

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True

# Type of vocoder, one of ['hifigan', 'vocos']
VOCODER_TYPE = 'vocos'
