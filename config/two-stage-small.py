# Configuration name
CONFIG = 'two-stage-small'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Discriminator loudness conditioning
DISCRIM_LOUDNESS_CONDITION = True

# Discriminator periodicity conditioning
DISCRIM_PERIODICITY_CONDITION = True

# Discriminator phoneme conditioning
DISCRIM_PHONEME_CONDITION = True

# Discriminator pitch conditioning
DISCRIM_PITCH_CONDITION = True

# Discriminator augmentation ratio conditioning
DISCRIM_RATIO_CONDITION = True

# Whether to perform gradient clipping on the generator
GRADIENT_CLIP_GENERATOR = 1000.

# Pass loudness through the latent
LATENT_LOUDNESS_SHORTCUT = True

# Pass periodicity through the latent
LATENT_PERIODICITY_SHORTCUT = True

# Pass the phonemes through the latent
LATENT_PHONEME_SHORTCUT = True

# Pass pitch through the latent
LATENT_PITCH_SHORTCUT = True

# Pass the augmentation ratio through the latent
LATENT_RATIO_SHORTCUT = True

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

# Whether to use snake activation in the audio generator
SNAKE = True

# Whether to use a two-stage model (text -> mels; mels -> waveform)
TWO_STAGE = True

# Reduce batch size and steps for development
BATCH_SIZE = 16
NUM_STEPS = 200000
