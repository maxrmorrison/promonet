MODULE = 'promonet'

# Configuration name
CONFIG = 'w2v2fb'

# Whether to perform speaker adaptation (instead of multi-speaker)
ADAPTATION = False

# Input features
INPUT_FEATURES = ['pitch', 'ppg']

# Number of training steps
NUM_STEPS = 500000

# Evaluation ratios for pitch-shifting, time-stretching, and loudness-scaling
EVALUATION_RATIOS = [.891, 1.12]
