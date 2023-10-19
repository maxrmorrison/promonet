MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-512bins'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

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
MODEL = 'end-to-end'

# Periodicity conditioning
PERIODICITY_FEATURES = True

# Pitch conditioning
PITCH_FEATURES = True

# Number of pitch bins
PITCH_BINS = 512

# Phonemic posteriorgram conditioning
PPG_FEATURES = True
