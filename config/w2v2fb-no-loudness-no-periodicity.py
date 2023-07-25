MODULE = 'promonet'

# Configuration name
CONFIG = 'w2v2fb-no-loudness-no-periodicity'

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

# Pitch conditioning
PITCH_FEATURES = True

# Phonemic posteriorgram conditioning
PPG_FEATURES = True
PPG_MODEL = 'w2v2fb'
PPG_CHANNELS = 42
