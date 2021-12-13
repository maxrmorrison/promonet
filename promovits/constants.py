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

# Location to save dataset partitions
PARTITION_DIR = ASSETS_DIR / 'partitions'

# Location to save training and adaptation artifacts
RUNS_DIR = Path(__file__).parent.parent / 'runs'

# Location to save adaptation artifacts
ADAPT_DIR = RUNS_DIR / 'adapt'

# Location to save adaptation artifacts
TRAIN_DIR = RUNS_DIR / 'train'

# Default checkpoint for inference
DEFAULT_CHECKPOINT = ASSETS_DIR / 'checkpoints' / 'promovits.pt'

# Default configuration file
DEFAULT_CONFIGURATION = ASSETS_DIR / 'configs' / 'promovits.py'


###############################################################################
# Audio parameters
###############################################################################


# Minimum and maximum frequency
FMIN = 50
FMAX = 550

# Audio hopsize
HOPSIZE = 256

# Maximum sample value of 16-bit audio
MAX_SAMPLE_VALUE = 32768

# Number of melspectrogram channels
NUM_MELS = 80

# Number of spectrogram channels
NUM_FFT = 1024

# Audio sample rate
SAMPLE_RATE = 22050

# Number of spectrogram channels
WINDOW_SIZE = 1024


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
# PPG parameters
###############################################################################


# Number of channels in the phonetic posteriorgram features
PPG_CHANNELS = 144


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
