MODULE = 'promonet'

# Configuration name
CONFIG = 'promonet-base-small'

# Loudness features
LOUDNESS_FEATURES = True

# Periodicity conditioning
PERIODICITY_FEATURES = True

# Pitch conditioning
PITCH_FEATURES = True

# Phonemic posteriorgram conditioning
PPG_FEATURES = True

# Reduce batch size and steps for development
BATCH_SIZE = 16
NUM_STEPS = 100000
