MODULE = 'promonet'

# Configuration name
CONFIG = 'conddisc-condgen'

# Condition discriminators on speech representation
CONDITION_DISCRIM = True

# Pass speech representation through the latent
LATENT_SHORTCUT = True

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

# Phonemic posteriorgram conditioning
PPG_FEATURES = True
