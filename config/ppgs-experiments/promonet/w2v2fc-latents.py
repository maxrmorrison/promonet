MODULE = 'promonet'

# Configuration name
CONFIG = 'w2v2fc-latents'

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
PPG_CHANNELS = 768

ADAPTATION=False

NUM_STEPS = 800000
