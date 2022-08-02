import torch

import promovits
from .modules import CausalConv1d, CausalTransposeConv1d


###############################################################################
# Model parameters
###############################################################################


# Whether to use causal or non-causal convolutions
CONV1D = CausalConv1d if promovits.CAUSAL else torch.nn.Conv1d
TRANSPOSECONV1D = \
    CausalTransposeConv1d if promovits.CAUSAL else torch.nn.ConvTranspose1d

# Hidden dimension channel sizes
HIDDEN_CHANNELS = 192
FILTER_CHANNELS = 768

# Speaker embedding size
GIN_CHANNELS = 256

# Convolutional kernel size
KERNEL_SIZE = 3

# (Negative) slope of leaky ReLU activations
LRELU_SLOPE = 0.1

# Number of attention heads
N_HEADS = 2

# Number of attention layers
N_LAYERS = 6

# Dropout probability
P_DROPOUT = 0.1


###############################################################################
# HiFi-GAN parameters
###############################################################################


# Kernel sizes of residual block
RESBLOCK_KERNEL_SIZES = [3, 7, 11]

# Dilation rates of residual block
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

# Upsample rates of residual blocks
UPSAMPLE_RATES = [8, 8, 2, 2]

# Initial channel size for upsampling layers
UPSAMPLE_INITIAL_SIZE = 512

# Kernel sizes of upsampling layers
UPSAMPLE_KERNEL_SIZES = [16, 16, 4, 4]
