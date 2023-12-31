MODULE = 'promonet'

# Configuration name
CONFIG = 'bottleneck-latents'

# Whether to perform speaker adaptation (instead of multi-speaker)
ADAPTATION = False

# Input features
INPUT_FEATURES = ['pitch', 'ppg']

# Number of training steps
NUM_STEPS = 500000

# Number of channels in the phonetic posteriorgram features
PPG_CHANNELS = 144

EVALUATION_RATIOS = [0.891, 1.122]
