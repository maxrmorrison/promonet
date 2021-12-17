import functools
from pathlib import Path

import torch


###############################################################################
# Directories
###############################################################################


# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent / 'assets'

# Location of preprocessed features
CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = Path(__file__).parent.parent / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = Path(__file__).parent.parent / 'eval'

# Location to save training and adaptation artifacts
RUNS_DIR = Path(__file__).parent.parent / 'runs'


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
# Features
###############################################################################


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
# Loss parameters
###############################################################################


# Weight applied to the melspectrogram loss
MEL_LOSS_WEIGHT = 45.


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

# Per-epoch decay rate of the learning rate
LEARNING_RATE_DECAY = .999875

# Number of training steps
NUM_STEPS = 300000

# Number of data loading worker threads
NUM_WORKERS = 2

# Seed for all random number generators
RANDOM_SEED = 1234

# Optimizer for training
TRAINING_OPTIMIZER = functools.partial(
    torch.optim.AdamW,
    lr=2e-4,
    betas=[0.8, 0.99],
    eps=1e-9)

# Number of samples generated during training
TRAINING_CHUNK_SIZE = 8192
