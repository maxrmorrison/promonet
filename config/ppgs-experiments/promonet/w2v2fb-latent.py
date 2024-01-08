MODULE = 'promonet'

# Configuration name
CONFIG = 'w2v2fb-latent'

# Whether to perform speaker adaptation (instead of multi-speaker)
ADAPTATION = False

# Batch size
BATCH_SIZE = 64

# Evaluation ratios for pitch-shifting, time-stretching, and loudness-scaling
EVALUATION_RATIOS = [.891, 1.12]

# Input features
INPUT_FEATURES = ['pitch', 'ppg']

# The model to use. One of ['hifigan', 'psola', 'vits', 'vocos', 'world'].
MODEL = 'vits'

# Number of training steps
STEPS = 250000

# Number of channels in the phonetic posteriorgram features
PPG_CHANNELS = 768

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest', 'slerp']
PPG_INTERP_METHOD = 'nearest'
