# Configuration name
CONFIG = 'test'

# Batch size (per gpu)
BATCH_SIZE = 16

# Number of training steps
NUM_STEPS = 100000

# Discriminator phoneme conditioning
DISCRIM_PHONEME_CONDITION = True

# Discriminator augmentation ratio conditioning
DISCRIM_RATIO_CONDITION = True

# Pass the phonemes through the latent
LATENT_PHONEME_SHORTCUT = False

# Pass the augmentation ratio through the latent
LATENT_RATIO_SHORTCUT = True


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

# Discriminator loudness conditioning
DISCRIM_LOUDNESS_CONDITION = True

# Discriminator periodicity conditioning
DISCRIM_PERIODICITY_CONDITION = True

# Discriminator pitch conditioning
DISCRIM_PITCH_CONDITION = True

# Whether to perform gradient clipping on the generator
GRADIENT_CLIP_GENERATOR = 1000.

# Pass loudness through the latent
LATENT_LOUDNESS_SHORTCUT = True

# Pass periodicity through the latent
LATENT_PERIODICITY_SHORTCUT = True

# Pass pitch through the latent
LATENT_PITCH_SHORTCUT = True

# Loudness features
LOUDNESS_FEATURES = True

# Whether to use the multi-resolution spectrogram discriminator from UnivNet
MULTI_RESOLUTION_DISCRIMINATOR = False

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = True

# Periodicity conditioning
PERIODICITY_FEATURES = True

# Pitch conditioning
PITCH_FEATURES = True

# Phonemic posteriorgram conditioning
PPG_FEATURES = True

# Type of interpolation method to use to scale PPG features
# Available method are ['nearest', 'linear']
PPG_INTERP_METHOD = 'nearest'

# Whether to use snake activation in the audio generator
SNAKE = True

# Whether to use a low-pass filter when using snake
SNAKE_FILTER = True
