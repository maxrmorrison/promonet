"""Config parameters whose values depend on other config parameters"""
import promovits


###############################################################################
# Directories
###############################################################################


# Location to save dataset partitions
PARTITION_DIR = promovits.ASSETS_DIR / 'partitions'

# Location to save adaptation artifacts
ADAPT_DIR = promovits.RUNS_DIR / 'adapt'

# Location to save adaptation artifacts
TRAIN_DIR = promovits.RUNS_DIR / 'train'

# Default checkpoint for inference
DEFAULT_CHECKPOINT = promovits.ASSETS_DIR / 'checkpoints' / 'promovits.pt'

# Default configuration file
DEFAULT_CONFIGURATION = promovits.ASSETS_DIR / 'configs' / 'promovits.py'


###############################################################################
# Features
###############################################################################


# Number of input features
# 178 is len(promovits.preprocess.text.symbols())
NUM_FEATURES = 178 if not promovits.PPG_FEATURES else (
    promovits.LOUDNESS_FEATURES +
    promovits.PERIODICITY_FEATURES +
    promovits.PITCH_FEATURES * promovits.PITCH_EMBEDDING_SIZE +
    promovits.PPG_FEATURES * promovits.PPG_CHANNELS)
