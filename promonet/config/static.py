import math

import promonet


###############################################################################
# Audio parameters
###############################################################################


# Threshold to sparsify Mel spectrograms
LOG_DYNAMIC_RANGE_COMPRESSION_THRESHOLD = (
    None if promonet.DYNAMIC_RANGE_COMPRESSION_THRESHOLD is None else
    math.log(promonet.DYNAMIC_RANGE_COMPRESSION_THRESHOLD))

# Base-2 log of pitch range boundaries
LOG_FMIN = math.log2(promonet.FMIN)
LOG_FMAX = math.log2(promonet.FMAX)


###############################################################################
# Directories
###############################################################################


# Location to save data augmentation information
AUGMENT_DIR = promonet.ASSETS_DIR / 'augmentations'

# Location to save dataset partitions
PARTITION_DIR = (
    promonet.ASSETS_DIR /
    'partitions' /
    ('adaptation' if promonet.ADAPTATION else 'multispeaker'))


###############################################################################
# Model parameters
###############################################################################


# Global input channels
GLOBAL_CHANNELS = (
    promonet.SPEAKER_CHANNELS +
    promonet.AUGMENT_PITCH +
    promonet.AUGMENT_LOUDNESS)

# Number of input features to the generator
NUM_FEATURES = promonet.NUM_MELS if promonet.SPECTROGRAM_ONLY else (
    promonet.PPG_CHANNELS +
    ('loudness' in promonet.INPUT_FEATURES) * promonet.LOUDNESS_BANDS +
    ('periodicity' in promonet.INPUT_FEATURES) +
    ('pitch' in promonet.INPUT_FEATURES) * (
        promonet.PITCH_EMBEDDING_SIZE if promonet.PITCH_EMBEDDING else 1))

# Number of input features to the discriminator
NUM_FEATURES_DISCRIM = 1

# Number of speakers
if promonet.TRAINING_DATASET == 'daps':
    NUM_SPEAKERS = 20
elif promonet.TRAINING_DATASET == 'libritts':
    NUM_SPEAKERS = 1230
elif promonet.TRAINING_DATASET == 'vctk':
    NUM_SPEAKERS = 109
else:
    raise ValueError(f'Dataset {promonet.TRAINING_DATASET} is not defined')

# Number of previous samples
if promonet.MODEL == 'cargan':
    NUM_PREVIOUS_SAMPLES = promonet.CARGAN_INPUT_SIZE
elif promonet.MODEL == 'fargan':
    NUM_PREVIOUS_SAMPLES = promonet.HOPSIZE * promonet.FARGAN_PREVIOUS_FRAMES
else:
    NUM_PREVIOUS_SAMPLES = 1
