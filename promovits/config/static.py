"""Config parameters whose values depend on other config parameters"""
import promovits


###############################################################################
# Directories
###############################################################################


# Location to save data augmentation information
AUGMENT_DIR = promovits.ASSETS_DIR / 'augmentations'

# Location to save dataset partitions
PARTITION_DIR = promovits.ASSETS_DIR / 'partitions'

# Default checkpoint for generation
DEFAULT_CHECKPOINT = promovits.ASSETS_DIR / 'checkpoints'

# Default configuration file
DEFAULT_CONFIGURATION = promovits.ASSETS_DIR / 'configs' / 'promovits.py'


###############################################################################
# Evaluation
###############################################################################


# Timer for benchmarking generation
TIMER = promovits.time.Context()


###############################################################################
# Features
###############################################################################


# Number of input features to the generator
# 178 is len(promovits.preprocess.text.symbols())
NUM_FEATURES = 178 if not promovits.PPG_FEATURES else (
    promovits.LOUDNESS_FEATURES +
    promovits.PERIODICITY_FEATURES +
    promovits.PITCH_FEATURES * promovits.PITCH_EMBEDDING_SIZE +
    promovits.PPG_FEATURES * promovits.PPG_CHANNELS)

# Number of input features to the discriminator
NUM_FEATURES_DISCRIM = (
    1 +
    promovits.DISCRIM_LOUDNESS_CONDITION +
    promovits.DISCRIM_PERIODICITY_CONDITION +
    promovits.DISCRIM_PITCH_CONDITION +
    promovits.DISCRIM_RATIO_CONDITION +
    promovits.DISCRIM_PHONEME_CONDITION * promovits.PPG_CHANNELS)

# Number of additional input features to the latent-to-audio model
ADDITIONAL_FEATURES_LATENT = (
    promovits.LATENT_PITCH_SHORTCUT * promovits.PITCH_EMBEDDING_SIZE +
    promovits.LATENT_LOUDNESS_SHORTCUT +
    promovits.LATENT_PERIODICITY_SHORTCUT +
    promovits.LATENT_PHONEME_SHORTCUT * promovits.PPG_CHANNELS +
    promovits.LATENT_RATIO_SHORTCUT +
    promovits.AUTOREGRESSIVE * promovits.AR_OUTPUT_SIZE)
