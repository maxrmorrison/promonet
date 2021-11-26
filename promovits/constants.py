from pathlib import Path


###############################################################################
# Directories
###############################################################################


# Location of preprocessed features
CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = Path(__file__).parent.parent / 'data' / 'datasets'

# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent / 'assets'

# Location to save dataset partitions
PARTITION_DIR = ASSETS_DIR / 'partitions'

# Location to save logs, checkpoints, and configurations
RUNS_DIR = Path(__file__).parent.parent / 'runs'

# Location to save evaluation artifacts
EVAL_DIR = Path(__file__).parent.parent / 'eval'

# Default checkpoint for inference
DEFAULT_CHECKPOINT = ASSETS_DIR / 'checkpoints' / 'promovits.pt'


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

# Number of data loading worker threads
NUM_WORKERS = 8

# Seed for all random number generators
RANDOM_SEED = 1234
