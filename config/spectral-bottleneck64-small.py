# Configuration name
CONFIG = 'spectral-bottleneck64-small'

# Only use spectral features
SPECTROGRAM_ONLY = True

# The size of the latent bottleneck
BOTTLENECK_SIZE = 64


###############################################################################
# Loss parameters
###############################################################################


# Weight applied to the discriminator loss
ADVERSARIAL_LOSS_WEIGHT = 2.

# Weight applied to the KL divergence loss
KL_DIVERGENCE_LOSS_WEIGHT = 1.

# Weight applied to the feature matching loss
FEATURE_MATCHING_LOSS_WEIGHT = 2.

# Weight applied to the melspectrogram loss
MEL_LOSS_WEIGHT = 45.


###############################################################################
# condition-both-mrd-snake-filter-clip-augment-small
###############################################################################


# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Whether to perform gradient clipping on the generator
GRADIENT_CLIP_GENERATOR = 1000.

# Whether to use the multi-resolution spectrogram discriminator from UnivNet
MULTI_RESOLUTION_DISCRIMINATOR = True

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = False

# Type of interpolation method to use to scale PPG features
# Available method are ['nearest', 'linear']
PPG_INTERP_METHOD = 'nearest'

# Whether to use snake activation in the audio generator
SNAKE = True

# Whether to use a low-pass filter when using snake
SNAKE_FILTER = True

# Reduce batch size and steps for development
BATCH_SIZE = 16
NUM_STEPS = 100000
