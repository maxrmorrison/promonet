MODULE = 'promonet'

# Configuration name
CONFIG = 'conddisc-condgen-augment-alignppg'

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
