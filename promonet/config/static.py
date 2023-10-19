import math

import pyfoal

import promonet


###############################################################################
# Directories
###############################################################################


# Location to save data augmentation information
AUGMENT_DIR = promonet.ASSETS_DIR / 'augmentations'

# Location to save dataset partitions
PARTITION_DIR = promonet.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = promonet.ASSETS_DIR / 'checkpoints'

# Default configuration file
DEFAULT_CONFIGURATION = promonet.ASSETS_DIR / 'configs' / 'promonet.py'


###############################################################################
# Audio parameters
###############################################################################


LOG_FMIN = math.log2(promonet.FMIN)
LOG_FMAX = math.log2(promonet.FMAX)

###############################################################################
# Model parameters
###############################################################################


# Global input channels
GLOBAL_CHANNELS = promonet.SPEAKER_CHANNELS + promonet.AUGMENT_PITCH

# Number of input features to the generator
NUM_FEATURES = (
    len(pyfoal.load.phonemes()) if not promonet.PPG_FEATURES else (
        promonet.LOUDNESS_FEATURES +
        promonet.PERIODICITY_FEATURES +
        promonet.PITCH_FEATURES * (
            promonet.PITCH_EMBEDDING_SIZE if promonet.PITCH_EMBEDDING else 1) +
        promonet.PPG_FEATURES * promonet.PPG_CHANNELS
    )
)

# Number of input features to the discriminator
NUM_FEATURES_DISCRIM = (
    1 +
    promonet.CONDITION_DISCRIM +
    promonet.CONDITION_DISCRIM +
    promonet.CONDITION_DISCRIM +
    promonet.CONDITION_DISCRIM * promonet.PPG_CHANNELS)

# Number of input features to the latent-to-audio model
if promonet.MODEL == 'hifigan' or promonet.MODEL == 'two-stage':
    LATENT_FEATURES = promonet.NUM_MELS
elif promonet.MODEL == 'vocoder':
    LATENT_FEATURES = NUM_FEATURES
else:
    LATENT_FEATURES = promonet.HIDDEN_CHANNELS + (
        promonet.LATENT_SHORTCUT * (
            promonet.PITCH_EMBEDDING_SIZE if promonet.PITCH_EMBEDDING else 1) +
        promonet.LATENT_SHORTCUT +
        promonet.LATENT_SHORTCUT +
        promonet.LATENT_SHORTCUT * promonet.PPG_CHANNELS
    )

# Number of speakers
if promonet.TRAINING_DATASET == 'daps':
    NUM_SPEAKERS = 20
elif promonet.TRAINING_DATASET == 'libritts':
    NUM_SPEAKERS = 1230
elif promonet.TRAINING_DATASET == 'vctk':
    NUM_SPEAKERS = 109
else:
    raise ValueError(f'Dataset {promonet.TRAINING_DATASET} is not defined')

# First stage of a two-stage model
TWO_STAGE = TWO_STAGE_1 = promonet.MODEL == 'two-stage'

# Second stage of a two-stage model
TWO_STAGE_2 = False
