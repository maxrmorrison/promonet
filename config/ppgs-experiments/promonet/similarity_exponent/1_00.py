MODULE = 'promonet'

# Configuration name
CONFIG = '1_00'

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
PPG_MODEL = 'w2v2fb-ppg'
PPG_CHANNELS = 40

ADAPTATION=False

NUM_STEPS = 800000

EVALUATION_RATIOS = [0.891, 1.122]