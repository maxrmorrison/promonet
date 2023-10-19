MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-noembed'

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

# Pitch embedding
PITCH_EMBEDDING = False

# Phonemic posteriorgram conditioning
PPG_FEATURES = True
