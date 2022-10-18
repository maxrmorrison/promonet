"""Config parameters whose values depend on other config parameters"""
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
# Features
###############################################################################


# First stage of a two-stage model
TWO_STAGE_1 = promonet.TWO_STAGE

# Second stage of a two-stage model
TWO_STAGE_2 = False

# Number of input features to the generator
# 178 is len(promonet.preprocess.text.symbols())
NUM_FEATURES = (
    178 if not promonet.PPG_FEATURES and not promonet.SPECTROGRAM_ONLY else (
        promonet.LOUDNESS_FEATURES +
        promonet.PERIODICITY_FEATURES +
        promonet.PITCH_FEATURES * (
            promonet.PITCH_EMBEDDING_SIZE if promonet.PITCH_EMBEDDING else 1) +
        (
            (promonet.NUM_FFT // 2 + 1) if promonet.SPECTROGRAM_ONLY else
            promonet.PPG_FEATURES * promonet.PPG_CHANNELS
        ) +
        256 * promonet.TEMPLATE_FEATURES
    )
)

# Number of input features to the discriminator
NUM_FEATURES_DISCRIM = (
    1 +
    promonet.DISCRIM_LOUDNESS_CONDITION +
    promonet.DISCRIM_PERIODICITY_CONDITION +
    promonet.DISCRIM_PITCH_CONDITION +
    promonet.DISCRIM_RATIO_CONDITION +
    promonet.DISCRIM_PHONEME_CONDITION * promonet.PPG_CHANNELS +
    promonet.TEMPLATE_RESIDUAL * 8)

# Number of additional input features to the latent-to-audio model
ADDITIONAL_FEATURES_LATENT = (
    promonet.LATENT_PITCH_SHORTCUT * (
        promonet.PITCH_EMBEDDING_SIZE if promonet.PITCH_EMBEDDING else 1) +
    promonet.LATENT_LOUDNESS_SHORTCUT +
    promonet.LATENT_PERIODICITY_SHORTCUT +
    promonet.LATENT_PHONEME_SHORTCUT * promonet.PPG_CHANNELS +
    promonet.LATENT_RATIO_SHORTCUT +
    promonet.AUTOREGRESSIVE * promonet.AR_OUTPUT_SIZE +
    promonet.SPECTROGRAM_ONLY * (promonet.NUM_FFT // 2 + 1))


###############################################################################
# Model parameters
###############################################################################


# Global input channels
GLOBAL_CHANNELS = promonet.SPEAKER_CHANNELS + (
    promonet.AUGMENT_PITCH and not promonet.SPECTROGRAM_ONLY)

# Hidden dimension channel sizes
HIDDEN_CHANNELS = promonet.BOTTLENECK_SIZE
