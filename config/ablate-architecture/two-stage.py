MODULE = 'promonet'

# Configuration name
CONFIG = 'two-stage'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Condition discriminators on speech representation
CONDITION_DISCRIM = False

# Pass speech representation through the latent
LATENT_SHORTCUT = False

# Loudness features
LOUDNESS_FEATURES = True

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
NUM_STEPS = 200000

# Number of adaptation steps
NUM_ADAPTATION_STEPS = 10000

# Periodicity conditioning
PERIODICITY_FEATURES = True

# Pitch conditioning
PITCH_FEATURES = True

# Phonemic posteriorgram conditioning
PPG_FEATURES = True
