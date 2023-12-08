MODULE = 'promonet'

# Configuration name
CONFIG = 'w2v2fb-ppg'

# Whether to perform speaker adaptation (instead of multi-speaker)
ADAPTATION = False

# Input features
INPUT_FEATURES = ['pitch', 'ppg']

# Number of training steps
NUM_STEPS = 250000

EVALUATION_RATIOS = [0.891, 1.122]