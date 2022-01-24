# Configuration name
CONFIG = 'ppg-nearest-pitch-periodicity-small'

# Periodicity conditioning
PERIODICITY_FEATURES = True

# Pitch conditioning
PITCH_FEATURES = True

# Phonemic posteriorgram conditioning
PPG_FEATURES = True

# Type of interpolation method to use to scale PPG features
# Available method are ['nearest', 'linear']
PPG_INTERP_METHOD = 'nearest'

# Reduce batch size and steps for development
BATCH_SIZE = 16
NUM_STEPS = 100000
