import functools
import os
from pathlib import Path

import torch


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'base'


###############################################################################
# Audio parameters
###############################################################################


# Minimum and maximum frequency
FMIN = 50.  # Hz
FMAX = 550.  # Hz

# Audio hopsize
HOPSIZE = 256  # samples

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


# Root location for saving outputs
# TEMPORARY
# ROOT_DIR = Path(__file__).parent.parent.parent
ROOT_DIR = Path('/files10/max/promonet')

# Location to save assets to be bundled with pip release
# TEMPORARY
# ASSETS_DIR = Path(__file__).parent.parent / 'assets'
ASSETS_DIR = Path('/files10/max/promonet/promonet/assets')

# Location of preprocessed features
CACHE_DIR = ROOT_DIR / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = ROOT_DIR / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = ROOT_DIR / 'eval'

# Location to save results
RESULTS_DIR = ROOT_DIR / 'results'

# Location to save training and adaptation artifacts
RUNS_DIR = ROOT_DIR / 'runs'

# Location of compressed datasets on disk
SOURCES_DIR = ROOT_DIR / 'data' / 'sources'


###############################################################################
# Data parameters
###############################################################################


# Whether to perform speaker adaptation (instead of multi-speaker)
ADAPTATION = True

# All features considered during preprocessing
ALL_FEATURES = [
    'alignment',
    'loudness',
    'periodicity',
    'pitch',
    'ppg',
    'spectrogram']

# Whether to use pitch augmentation
AUGMENT_PITCH = False

# Maximum ratio for pitch augmentation
AUGMENTATION_RATIO_MAX = 2.

# Minimum ratio for pitch augmentation
AUGMENTATION_RATIO_MIN = .5

# Condition discriminators on speech representation
CONDITION_DISCRIM = False

# Names of all datasets
DATASETS = ['daps', 'libritts', 'vctk']

# Pass speech representation through the latent
LATENT_SHORTCUT = False

# Whether to use an embedding layer for pitch
PITCH_EMBEDDING = True

# Number of pitch bins
PITCH_BINS = 256

# Embedding size used to represent each pitch bin
PITCH_EMBEDDING_SIZE = 64

# Number of channels in the phonetic posteriorgram features
PPG_CHANNELS = 40

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest', 'slerp']
PPG_INTERP_METHOD = 'nearest'

# Seed for all random number generators
RANDOM_SEED = 1234

# Only use spectral features
SPECTROGRAM_ONLY = False

# Dataset to use for training
TRAINING_DATASET = 'vctk'


###############################################################################
# Evaluation parameters
###############################################################################


# Evaluation ratios for pitch-shifting, time-stretching, and loudness-scaling
EVALUATION_RATIOS = [.717, 1.414]


###############################################################################
# Logging parameters
###############################################################################


# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 20000  # steps

# Number of steps between logging to Tensorboard
LOG_INTERVAL = 2500  # steps


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
# Model parameters
###############################################################################


# The size of the latent bottleneck
HIDDEN_CHANNELS = 192

# Input features
INPUT_FEATURES = ['loudness', 'periodicity', 'pitch', 'ppg']

# Hidden dimension channel size
FILTER_CHANNELS = 768

# Convolutional kernel size
KERNEL_SIZE = 3

# (Negative) slope of leaky ReLU activations
LRELU_SLOPE = .1

# The model to use. One of [
#     'end-to-end',
#     'hifigan',
#     'psola',
#     'two-stage',
#     'vits',
#     'vocoder',
#     'world'
# ]
MODEL = 'end-to-end'

# Whether to use the multi-resolution spectrogram discriminator from UnivNet
MULTI_RESOLUTION_DISCRIMINATOR = False

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = True

# Number of attention heads
N_HEADS = 2

# Number of attention layers
N_LAYERS = 6

# Noise scales for inference
NOISE_SCALE_INFERENCE = .667
NOISE_SCALE_W_INFERENCE = .8

# Dropout probability
P_DROPOUT = .1

# Kernel sizes of residual block
RESBLOCK_KERNEL_SIZES = [3, 7, 11]

# Dilation rates of residual block
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

# Whether to use snake activation in the audio generator
SNAKE = False

# Speaker embedding size
SPEAKER_CHANNELS = 256

# Initial channel size for upsampling layers
UPSAMPLE_INITIAL_SIZE = 512

# Kernel sizes of upsampling layers
UPSAMPLE_KERNEL_SIZES = [16, 16, 4, 4]

# Upsample rates of residual blocks
UPSAMPLE_RATES = [8, 8, 2, 2]

# Whether to omit latent generation
VOCODER = False

# Type of vocoder, one of ['hifigan', 'vocos']
VOCODER_TYPE = 'hifigan'


###############################################################################
# Training parameters
###############################################################################


# Number of items in a batch
BATCH_SIZE = 32

# Number of buckets to partition training and validation data into based on
# length to avoid excess padding
BUCKETS = 8

# Maximum length of frames during training
MAX_FRAME_LENGTH = 1600

# Maximum length of phonemes during training
MAX_TEXT_LENGTH = 190

# Number of samples generated during training
CHUNK_SIZE = 8192

# Gradients with norms above this value are clipped to this value
GRADIENT_CLIP_GENERATOR = None

# Number of training steps
NUM_STEPS = 200000

# Number of adaptation steps
NUM_ADAPTATION_STEPS = 10000

# Number of data loading worker threads
NUM_WORKERS = os.cpu_count() // 4

# Training optimizer
OPTIMIZER = functools.partial(
    torch.optim.AdamW,
    lr=2e-4,
    betas=(.8, .99),
    eps=1e-9)
