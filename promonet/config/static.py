import math

import promonet


###############################################################################
# Audio parameters
###############################################################################


# Base-2 log of pitch range boundaries
LOG_FMIN = math.log2(promonet.FMIN)
LOG_FMAX = math.log2(promonet.FMAX)


###############################################################################
# Directories
###############################################################################


# Location to save data augmentation information
AUGMENT_DIR = promonet.ASSETS_DIR / 'augmentations'

# Location to save dataset partitions
PARTITION_DIR = promonet.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = promonet.ASSETS_DIR / 'checkpoints'


###############################################################################
# Evaluation parameters
###############################################################################


# Features to plot
DEFAULT_PLOT_FEATURES = ['audio'] + promonet.INPUT_FEATURES


###############################################################################
# Model parameters
###############################################################################


# Global input channels
GLOBAL_CHANNELS = promonet.SPEAKER_CHANNELS + promonet.AUGMENT_PITCH

# Number of input features to the generator
NUM_FEATURES = (
    promonet.PPG_CHANNELS +
    ('loudness' in promonet.INPUT_FEATURES) +
    ('periodicity' in promonet.INPUT_FEATURES) +
    ('pitch' in promonet.INPUT_FEATURES) * (
        promonet.PITCH_EMBEDDING_SIZE if promonet.PITCH_EMBEDDING else 1))

# Number of input features to the discriminator
NUM_FEATURES_DISCRIM = (
    1 +
    promonet.CONDITION_DISCRIM +
    promonet.CONDITION_DISCRIM +
    promonet.CONDITION_DISCRIM +
    promonet.CONDITION_DISCRIM * promonet.PPG_CHANNELS)

# Number of speakers
if promonet.TRAINING_DATASET == 'daps':
    NUM_SPEAKERS = 20
elif promonet.TRAINING_DATASET == 'libritts':
    NUM_SPEAKERS = 1230
elif promonet.TRAINING_DATASET == 'vctk':
    NUM_SPEAKERS = 109
else:
    raise ValueError(f'Dataset {promonet.TRAINING_DATASET} is not defined')
