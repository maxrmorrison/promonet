MODULE = 'promonet'

# Configuration name
CONFIG = 'w2v2fc-latent'

# Whether to perform speaker adaptation (instead of multi-speaker)
ADAPTATION = False

# Evaluation ratios for pitch-shifting, time-stretching, and loudness-scaling
EVALUATION_RATIOS = [.891, 1.12]

# Input features
INPUT_FEATURES = ['pitch', 'ppg']

# Number of training steps
STEPS = 500000

# Number of channels in the phonetic posteriorgram features
PPG_CHANNELS = 768

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest', 'slerp']
PPG_INTERP_METHOD = 'nearest'
