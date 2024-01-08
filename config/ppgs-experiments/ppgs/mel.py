MODULE = 'ppgs'

# Configuration name
CONFIG = 'mel'

# Network width
HIDDEN_CHANNELS = 256

# Dimensionality of input representation
INPUT_CHANNELS = 80

# Number of hidden layers
NUM_HIDDEN_LAYERS = 5

# Input representation
REPRESENTATION = 'mel'

# Local checkpoint to use
# If None, Huggingface will be used unless a checkpoint is given in the CLI
LOCAL_CHECKPOINT = f'/repos/ppgs/runs/{CONFIG}/00200000.pt'
