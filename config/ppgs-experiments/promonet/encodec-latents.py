MODULE = 'promonet'

# Configuration name
CONFIG = 'encodec-latents'

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
PPG_MODEL = 'encodec-latents'
PPG_CHANNELS = 128

ADAPTATION=False