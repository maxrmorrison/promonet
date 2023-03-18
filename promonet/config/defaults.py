import functools
from pathlib import Path

import torch


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'promonet'

# Module name
MODULE = 'promonet'


###############################################################################
# Audio parameters
###############################################################################


# Minimum and maximum frequency
FMIN = 50.  # Hz
FMAX = 550.  # Hz

# Audio hopsize
HOPSIZE = 256  # samples

# Maximum sample value of 16-bit audio
MAX_SAMPLE_VALUE = 32768

# Number of melspectrogram channels
NUM_MELS = 80

# Number of spectrogram channels
NUM_FFT = 1024

# Audio sample rate
SAMPLE_RATE = 22050  # Hz

# Number of spectrogram channels
WINDOW_SIZE = 1024

###############################################################################
# Data parameters
###############################################################################


# Number of buckets to partition training and validation data into based on
# length to avoid excess padding
# TODO
BUCKETS = 8

# Names of all datasets
DATASETS = ['daps', 'vctk']

# Datasets for evaluation
EVALUATION_DATASETS = DATASETS


###############################################################################
# Directories
###############################################################################


# Root location for saving outputs
# TEMPORARY
# ROOT_DIR = Path(__file__).parent.parent.parent
ROOT_DIR = Path('/data/max/promonet')

# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of preprocessed features
CACHE_DIR = ROOT_DIR / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = ROOT_DIR / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = ROOT_DIR / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = ROOT_DIR / 'runs'

# Location of compressed datasets on disk
SOURCES_DIR = ROOT_DIR / 'data' / 'sources'


###############################################################################
# Evaluation
###############################################################################


# Whether to perform benchmarking
BENCHMARK = False

# The model to use for evaluation.
# One of ['promonet', 'psola', 'world].
MODEL = 'promonet'


###############################################################################
# Features
###############################################################################


# Whether to use pitch augmentation
AUGMENT_PITCH = False

# Discriminator loudness conditioning
DISCRIM_LOUDNESS_CONDITION = False

# Discriminator periodicity conditioning
DISCRIM_PERIODICITY_CONDITION = False

# Discriminator pitch conditioning
DISCRIM_PITCH_CONDITION = False

# Discriminator phoneme conditioning
DISCRIM_PHONEME_CONDITION = False

# Discriminator augmentation ratio conditioning
DISCRIM_RATIO_CONDITION = False

# Pass loudness through the latent
LATENT_LOUDNESS_SHORTCUT = False

# Pass periodicity through the latent
LATENT_PERIODICITY_SHORTCUT = False

# Pass pitch through the latent
LATENT_PITCH_SHORTCUT = False

# Pass the phonemes through the latent
LATENT_PHONEME_SHORTCUT = False

# Pass the augmentation ratio through the latent
LATENT_RATIO_SHORTCUT = False

# A-weighted loudness conditioning
LOUDNESS_FEATURES = False

# Periodicity conditioning
PERIODICITY_FEATURES = False

# Whether to use an embedding layer for pitch
PITCH_EMBEDDING = True

# Pitch conditioning
PITCH_FEATURES = False

# Number of pitch bins
PITCH_BINS = 256

# Embedding size used to represent each pitch bin
PITCH_EMBEDDING_SIZE = 64

# Number of channels in the phonetic posteriorgram features
PPG_CHANNELS = 144

# Phonemic posteriorgram conditioning
PPG_FEATURES = False

# Type of interpolation method to use to scale PPG features
# Available method are ['nearest', 'linear']
PPG_INTERP_METHOD = None

# Type of PPGs to use
PPG_MODEL = None

# Only use spectral features
SPECTROGRAM_ONLY = False

# Whether to use a two-stage model
TWO_STAGE = False


###############################################################################
# Logging parameters
###############################################################################


# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 25000  # steps

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 1000  # steps

# Number of steps between evaluation
EVALUATION_INTERVAL = 2500  # steps


###############################################################################
# Model parameters
###############################################################################


# Whether to use autoregression
AUTOREGRESSIVE = False

# The size of autoregressive embedding layers
AR_HIDDEN_SIZE = 256

# The number of previous samples to use for autoregression
AR_INPUT_SIZE = 512

# The size of the output autoregressive embedding
AR_OUTPUT_SIZE = 128

# The size of the latent bottleneck
BOTTLENECK_SIZE = 192

# Whether to use causal layers
CAUSAL = False

# Hidden dimension channel size
FILTER_CHANNELS = 768

# Convolutional kernel size
KERNEL_SIZE = 3

# (Negative) slope of leaky ReLU activations
LRELU_SLOPE = 0.1

# Whether to use the multi-resolution spectrogram discriminator from UnivNet
MULTI_RESOLUTION_DISCRIMINATOR = False

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = True

# Number of attention heads
N_HEADS = 2

# Number of attention layers
N_LAYERS = 6

# Dropout probability
P_DROPOUT = 0.1

# Kernel sizes of residual block
RESBLOCK_KERNEL_SIZES = [3, 7, 11]

# Dilation rates of residual block
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

# Whether to use snake activation in the audio generator
SNAKE = False

# Whether to use exact filter values as BigVGan
SNAKE_EXACT = False

# Whether to use a low-pass filter when using snake
SNAKE_FILTER = False

# Speaker embedding size
SPEAKER_CHANNELS = 256

# Whether to use a T5X stack for phoneme encoding
T5_ENCODER = False

# Initial channel size for upsampling layers
UPSAMPLE_INITIAL_SIZE = 512

# Kernel sizes of upsampling layers
UPSAMPLE_KERNEL_SIZES = [16, 16, 4, 4]

# Upsample rates of residual blocks
UPSAMPLE_RATES = [8, 8, 2, 2]

# Whether to omit latent generation
VOCODER = False


###############################################################################
# Loss parameters
###############################################################################


# Weight applied to the discriminator loss
ADVERSARIAL_LOSS_WEIGHT = 1.

# Weight applied to the KL divergence loss
KL_DIVERGENCE_LOSS_WEIGHT = 1.

# Weight applied to the feature matching loss
FEATURE_MATCHING_LOSS_WEIGHT = 1.

# Whether to omit the first activation of each discriminator
FEATURE_MATCHING_OMIT_FIRST = False

# Weight applied to the melspectrogram loss
MEL_LOSS_WEIGHT = 45.


###############################################################################
# Optimizers
###############################################################################


# Optimizer for training
OPTIMIZER = functools.partial(
    torch.optim.AdamW,
    lr=2e-4,
    betas=[0.8, 0.99],
    eps=1e-9)


###############################################################################
# Text parameters
###############################################################################


# Minimum length of text input
MIN_TEXT_LEN = 1

# Maximum length of text input
MAX_TEXT_LEN = 190


###############################################################################
# Training parameters
###############################################################################


# Batch size (per gpu)
BATCH_SIZE = 64

# Number of samples generated during training
CHUNK_SIZE = 8192

# Whether to perform gradient clipping on the generator
GRADIENT_CLIP_GENERATOR = None

# Per-epoch decay rate of the learning rate
LEARNING_RATE_DECAY = .999875

# Number of training steps
NUM_STEPS = 300000

# Number of adaptation steps
NUM_ADAPTATION_STEPS = 5000

# Number of data loading worker threads
NUM_WORKERS = 2

# Seed for all random number generators
RANDOM_SEED = 1234

# Aligner to use to evaluate training
TRAIN_ALIGNER = 'p2fa'
