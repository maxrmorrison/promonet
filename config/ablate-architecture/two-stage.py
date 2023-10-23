MODULE = 'promonet'

# Configuration name
CONFIG = 'two-stage'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# The model to use. One of [
#     'end-to-end',
#     'hifigan',
#     'psola',
#     'two-stage',
#     'vits',
#     'vocoder',
#     'world'
# ]
MODEL = 'two-stage'

# Number of training steps
NUM_STEPS = 400000

# Number of adaptation steps
NUM_ADAPTATION_STEPS = 20000
