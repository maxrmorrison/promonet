import functools
from pathlib import Path

import torch


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'promovits'


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
# Directories
###############################################################################


# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of preprocessed features
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = Path(__file__).parent.parent.parent / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = Path(__file__).parent.parent.parent / 'runs'


###############################################################################
# Evaluation
###############################################################################


# Whether to perform benchmarking
BENCHMARK = False

# The model to use for evaluation.
# One of ['promovits', 'promospec', 'qpcargan', 'clpcnet', 'psola', 'world].
MODEL = 'promovits'


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

# Pass loudness through the latent
LATENT_LOUDNESS_SHORTCUT = False

# Pass periodicity through the latent
LATENT_PERIODICITY_SHORTCUT = False

# Pass pitch through the latent
LATENT_PITCH_SHORTCUT = False

# A-weighted loudness conditioning
LOUDNESS_FEATURES = False

# Periodicity conditioning
PERIODICITY_FEATURES = False

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

# Whether to use causal layers for producing the latent
CAUSAL = False

# Whether to use the multi-resolution spectrogram discriminator from UnivNet
MULTI_RESOLUTION_DISCRIMINATOR = False

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = True


###############################################################################
# Loss parameters
###############################################################################


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
