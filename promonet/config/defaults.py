import functools
import os
from pathlib import Path

import GPUtil
import torch


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'promonet'


###############################################################################
# Audio parameters
###############################################################################


# Threshold to sparsify Mel spectrograms
DYNAMIC_RANGE_COMPRESSION_THRESHOLD = None

# Minimum and maximum frequency
FMIN = 50.  # Hz
FMAX = 550.  # Hz

# Audio hopsize
HOPSIZE = 256  # samples

# Maximum number of speech harmonics
MAX_HARMONICS = 3

# Minimum decibel level
MIN_DB = -100.

# Number of melspectrogram channels
NUM_MELS = 80

# Number of spectrogram channels
NUM_FFT = 1024

# Reference decibel level
REF_DB = 20.

# Audio sample rate
SAMPLE_RATE = 22050  # Hz

# Number of spectrogram channels
WINDOW_SIZE = 1024


###############################################################################
# Data parameters
###############################################################################


# Whether to perform speaker adaptation (instead of multi-speaker)
ADAPTATION = False

# All features considered during preprocessing
ALL_FEATURES = [
    'loudness',
    'pitch',
    'periodicity',
    'ppg',
    'spectrogram',
    'text',
    'harmonics',
    'speaker']

# Whether to use loudness augmentation
AUGMENT_LOUDNESS = True

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Maximum ratio for pitch augmentation
AUGMENTATION_RATIO_MAX = 2.

# Minimum ratio for pitch augmentation
AUGMENTATION_RATIO_MIN = .5

# Names of all datasets
DATASETS = ['daps', 'libritts', 'vctk']

# Number of bands of A-weighted loudness
LOUDNESS_BANDS = 8

# Whether to use an embedding layer for pitch
PITCH_EMBEDDING = True

# Number of pitch bins
PITCH_BINS = 256

# Embedding size used to represent each pitch bin
PITCH_EMBEDDING_SIZE = 64

# Number of channels in the phonetic posteriorgram features
PPG_CHANNELS = 40

# Type of interpolation method to use to scale PPG features
# Available method are ['linear', 'nearest']
PPG_INTERP_METHOD = 'linear'

# Whether to shift Mel inputs to have a minimum of zero
SPARSE_MELS = False

# Type of sparsification used for ppgs
# One of ['constant', 'percentile', 'topk', None]
SPARSE_PPG_METHOD = 'percentile'

# Threshold for ppg sparsification.
# In [0, 1] for 'contant' and 'percentile'; integer > 0 for 'topk'.
SPARSE_PPG_THRESHOLD = 0.85

# Seed for all random number generators
RANDOM_SEED = 1234

# Only use spectral features
SPECTROGRAM_ONLY = False

# Dataset to use for training
TRAINING_DATASET = 'vctk'

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True

# Whether to perform Viterbi decoding on pitch features
VITERBI_DECODE_PITCH = True

# Default periodicity threshold of the voiced/unvoiced decision
VOICING_THRESHOLD = .1625


###############################################################################
# Directories
###############################################################################


# Root location for saving outputs
ROOT_DIR = Path(__file__).parent.parent.parent

# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

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


###############################################################################
# Discriminator parameters
###############################################################################


# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Whether to use the multi-period waveform discriminator from HiFi-GAN
MULTI_PERIOD_DISCRIMINATOR = True

# Whether to use the multi-resolution spectrogram discriminator from UnivNet
MULTI_RESOLUTION_DISCRIMINATOR = False

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = False


###############################################################################
# Evaluation parameters
###############################################################################


# Features to plot
DEFAULT_PLOT_FEATURES = ['audio', 'loudness', 'pitch', 'periodicity', 'ppg']

# Error threshold beyond which a frame of loudness is considered incorrect
ERROR_THRESHOLD_LOUDNESS = 6.  # decibels

# Error threshold beyond which a frame of periodicity is considered incorrect
ERROR_THRESHOLD_PERIODICITY = .1

# Error threshold beyond which a frame of pitch is considered incorrect
ERROR_THRESHOLD_PITCH = 50.  # cents

# Error threshold beyond which a frame of PPG is considered incorrect
ERROR_THRESHOLD_PPG = .1  # JSD

# Evaluation ratios for pitch-shifting, time-stretching, and loudness-scaling
EVALUATION_RATIOS = [.717, 1.414]


###############################################################################
# Generator parameters
###############################################################################


# Input features
INPUT_FEATURES = ['loudness', 'pitch', 'periodicity', 'ppg']

# (Negative) slope of leaky ReLU activations
LRELU_SLOPE = .1

# The model to use.
# One of ['cargan', 'fargan', 'hifigan', 'vocos', 'world'].
MODEL = 'hifigan'

# Number of previous samples to use
CARGAN_INPUT_SIZE = 2 * HOPSIZE

# Autoregressive hidden size
CARGAN_HIDDEN_SIZE = 256

# Number of autoregressive output channels
CARGAN_OUTPUT_SIZE = 128

# Whether to use additive noise with FARGAN
FARGAN_ADDITIVE_NOISE = True

# Whether to use the same discriminator as FARGAN
FARGAN_DISCRIMINATOR = False

# Whether to use gain normalization in the subframe network
FARGAN_GAIN_NORMALIZATION = False

# Number of previous frames used for lookback in FARGAN
FARGAN_PREVIOUS_FRAMES = 2  # frames

# Number of subframes per frame
FARGAN_SUBFRAMES = 4  # subframes

# Number of samples per subframe
FARGAN_SUBFRAME_SIZE = HOPSIZE // FARGAN_SUBFRAMES  # samples

# Kernel sizes of residual block
HIFIGAN_RESBLOCK_KERNEL_SIZES = [3, 7, 11]

# Dilation rates of residual block
HIFIGAN_RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

# Initial channel size for upsampling layers
HIFIGAN_UPSAMPLE_INITIAL_SIZE = 512

# Kernel sizes of upsampling layers
HIFIGAN_UPSAMPLE_KERNEL_SIZES = [16, 16, 4, 4]

# Upsample rates of residual blocks
HIFIGAN_UPSAMPLE_RATES = [8, 8, 2, 2]

# Speaker embedding size
SPEAKER_CHANNELS = 256

# The size of intermediate feature activations in VITS
VITS_CHANNELS = 192

# Hidden dimension channel size
VITS_PRIOR_CHANNELS = 768

# The size of feature activations in Vocos
VOCOS_CHANNELS = 512

# The size of pointwise convolutions in Vocos
VOCOS_POINTWISE_CHANNELS = 1536

# Number of neural network layers in Vocos
VOCOS_LAYERS = 6

# Number of channels of WavLM x-vector embedding
WAVLM_EMBEDDING_CHANNELS = 512

# Whether to use WavLM x-vectors for zero-shot speaker conditioning
ZERO_SHOT = False

# Whether to shuffle speaker embeddings during training
ZERO_SHOT_SHUFFLE = False


###############################################################################
# Logging parameters
###############################################################################


# Number of steps between saving checkpoints
CHECKPOINT_INTERVAL = 20000  # steps

# Number of steps between logging to Tensorboard
EVALUATION_INTERVAL = 2500  # steps

# Number of steps to perform for tensorboard logging
DEFAULT_EVALUATION_STEPS = 16

# Number of examples to plot while evaluating during training
PLOT_EXAMPLES = 10


###############################################################################
# Loss parameters
###############################################################################


# Whether to use hinge loss instead of L2
ADVERSARIAL_HINGE_LOSS = False

# Step to start using adversarial loss
ADVERSARIAL_LOSS_START_STEP = 0

# Weight applied to the discriminator loss
ADVERSARIAL_LOSS_WEIGHT = 1.

# Step to start training discriminator
DISCRIMINATOR_START_STEP = 0

# Weight applied to the feature matching loss
FEATURE_MATCHING_LOSS_WEIGHT = 1.

# Whether to omit the first activation of each discriminator
FEATURE_MATCHING_OMIT_FIRST = False

# Weight applied to the KL divergence loss
KL_DIVERGENCE_LOSS_WEIGHT = 1.

# Whether to use mel spectrogram loss
MEL_LOSS = True

# Weight applied to the melspectrogram loss
MEL_LOSS_WEIGHT = 45.

# Whether to use multi-mel loss
MULTI_MEL_LOSS = False

# Window sizes to be used in the multi-scale mel loss
MULTI_MEL_LOSS_WINDOWS = [32, 64, 128, 256, 512, 1024, 2048]

# Whether to compare raw audio signals
SIGNAL_LOSS = False

# Weight applied to signal loss
SIGNAL_LOSS_WEIGHT = .03

# Whether to shift the Mels given to the Mel loss to have a minimum of zero
SPARSE_MEL_LOSS = False

# Whether to use multi-resolution spectral convergence loss
SPECTRAL_CONVERGENCE_LOSS = False


###############################################################################
# Training parameters
###############################################################################


# Batch size
BATCH_SIZE = 64

# Training sequence length
CHUNK_SIZE = 16384  # samples

# Gradients above this value are clipped to this value
GRADIENT_CLIP_GENERATOR = None

# Number of training steps
STEPS = 800000

# Number of adaptation steps
ADAPTATION_STEPS = 10000

# Number of data loading worker threads
# TEMPORARY
# try:
#     NUM_WORKERS = int(os.cpu_count() / max(1, len(GPUtil.getGPUs())))
# except ValueError:
#     NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 10

# Training optimizer
OPTIMIZER = functools.partial(
    torch.optim.AdamW,
    lr=2e-4,
    betas=(.8, .99),
    eps=1e-9)
