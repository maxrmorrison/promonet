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
# Evaluation
###############################################################################


# Timer for benchmarking generation
TIMER = promonet.time.Context()


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
    promonet.DISCRIM_LOUDNESS_CONDITION +
    promonet.DISCRIM_PERIODICITY_CONDITION +
    promonet.DISCRIM_PITCH_CONDITION +
    promonet.DISCRIM_PHONEME_CONDITION * promonet.PPG_CHANNELS)

# Number of input features to the latent-to-audio model
if promonet.MODEL == 'hifigan' or promonet.MODEL == 'two-stage':
    LATENT_FEATURES = promonet.NUM_MELS
elif promonet.MODEL == 'vocoder':
    LATENT_FEATURES = NUM_FEATURES
else:
    LATENT_FEATURES = promonet.HIDDEN_CHANNELS + (
        promonet.LATENT_PITCH_SHORTCUT * (
            promonet.PITCH_EMBEDDING_SIZE if promonet.PITCH_EMBEDDING else 1) +
        promonet.LATENT_LOUDNESS_SHORTCUT +
        promonet.LATENT_PERIODICITY_SHORTCUT +
        promonet.LATENT_PHONEME_SHORTCUT * promonet.PPG_CHANNELS
    )

# First stage of a two-stage model
TWO_STAGE = TWO_STAGE_1 = promonet.MODEL == 'two-stage'

# Second stage of a two-stage model
TWO_STAGE_2 = False
