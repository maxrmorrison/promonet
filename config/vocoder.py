MODULE = 'promonet'

# Configuration name
CONFIG = 'vocoder'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Discriminator phoneme conditioning
DISCRIM_PHONEME_CONDITION = True

# Discriminator loudness conditioning
DISCRIM_LOUDNESS_CONDITION = True

# Discriminator periodicity conditioning
DISCRIM_PERIODICITY_CONDITION = True

# Discriminator pitch conditioning
DISCRIM_PITCH_CONDITION = True

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
