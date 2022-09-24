import torch

import promonet
from .modules import CausalConv1d, CausalTransposeConv1d


###############################################################################
# Model parameters
###############################################################################


# Whether to use causal or non-causal convolutions
CONV1D = CausalConv1d if promonet.CAUSAL else torch.nn.Conv1d
TRANSPOSECONV1D = \
    CausalTransposeConv1d if promonet.CAUSAL else torch.nn.ConvTranspose1d
