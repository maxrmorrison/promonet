# TODO - document parameters
INTER_CHANNELS = 192
HIDDEN_CHANNELS = 192
FILTER_CHANNELS = 768

# Speaker embedding size
GIN_CHANNELS = 256

# Convolutional kernel size
KERNEL_SIZE = 3

# Number of attention heads
N_HEADS = 2

# Number of attention layers
N_LAYERS = 6

# Dropout probability
P_DROPOUT = 0.1

RESBLOCK = 1
RESBLOCK_KERNEL_SIZES = [3, 7, 11]
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
UPSAMPLE_RATES = [8, 8, 2, 2]
UPSAMPLE_INITIAL_COUNT = 512
UPSAMPLE_KERNEL_SIZES = [16, 16, 4, 4]
N_LAYERS_Q = 3
