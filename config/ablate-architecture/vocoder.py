MODULE = 'promonet'

# Configuration name
CONFIG = 'vocoder'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Discriminator phoneme conditioning
CONDITION_DISCRIM = True

# Discriminator loudness conditioning
CONDITION_DISCRIM = True

# Discriminator periodicity conditioning
CONDITION_DISCRIM = True

# Discriminator pitch conditioning
CONDITION_DISCRIM = True

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
MODEL = 'vocoder'

# Periodicity conditioning
PERIODICITY_FEATURES = True

# Pitch conditioning
PITCH_FEATURES = True

# Phonemic posteriorgram conditioning
PPG_FEATURES = True
