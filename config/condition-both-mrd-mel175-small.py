# Configuration name
CONFIG = 'condition-both-mrd-mel175-small'

# Discriminator loudness conditioning
DISCRIM_LOUDNESS_CONDITION = True

# Discriminator periodicity conditioning
DISCRIM_PERIODICITY_CONDITION = True

# Discriminator pitch conditioning
DISCRIM_PITCH_CONDITION = True

# Pass loudness through the latent
LATENT_LOUDNESS_SHORTCUT = True

# Pass periodicity through the latent
LATENT_PERIODICITY_SHORTCUT = True

# Pass pitch through the latent
LATENT_PITCH_SHORTCUT = True

# Loudness features
LOUDNESS_FEATURES = True

# Weight applied to the melspectrogram loss
MEL_LOSS_WEIGHT = 17.5

# Whether to use the multi-resolution spectrogram discriminator from UnivNet
MULTI_RESOLUTION_DISCRIMINATOR = True

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = False

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