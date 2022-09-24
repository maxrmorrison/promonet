# Configuration name
CONFIG = 'ppg-nearest-pitch-periodicity-loudness-small'

# Batch size (per gpu)
BATCH_SIZE = 24

# Number of training steps
NUM_STEPS = 250000

# Loudness features
LOUDNESS_FEATURES = True

# Periodicity conditioning
PERIODICITY_FEATURES = True

# Pitch conditioning
PITCH_FEATURES = True

# Phonemic posteriorgram conditioning
PPG_FEATURES = True

# Type of interpolation method to use to scale PPG features
# Available method are ['nearest', 'linear']
PPG_INTERP_METHOD = 'nearest'
