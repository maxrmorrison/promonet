MODULE = 'promonet'

# Configuration name
CONFIG = 'bottleneck-latents'

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
PPG_MODEL = 'bottleneck-latents'
PPG_CHANNELS = 144

ADAPTATION=False

NUM_STEPS = 800000