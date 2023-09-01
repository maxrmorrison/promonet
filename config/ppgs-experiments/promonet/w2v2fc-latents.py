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
PITCH_EVAL_METHOD = 'cents'

# Phonemic posteriorgram conditioning
PPG_FEATURES = True
PPG_MODEL = 'w2v2fc-latents'
PPG_CHANNELS = 768

ADAPTATION=False