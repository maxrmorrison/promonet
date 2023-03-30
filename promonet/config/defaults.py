from pathlib import Path


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'promonet'


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
# Data parameters
###############################################################################


# Whether to use pitch augmentation
AUGMENT_PITCH = False

# Maximum ratio for pitch augmentation
AUGMENTATION_RATIO_MAX = 2.

# Minimum ratio for pitch augmentation
AUGMENTATION_RATIO_MIN = .5

# Names of all datasets
DATASETS = ['daps', 'vctk']

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

# Maximum length of text input
MAX_TEXT_LEN = 190

# Minimum length of text input
MIN_TEXT_LEN = 1

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
# TODO - deprecate in favor of aligned features
PPG_INTERP_METHOD = 'nearest'

# Type of PPGs to use
PPG_MODEL = None

# Only use spectral features
SPECTROGRAM_ONLY = False


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

# Location to save training and adaptation artifacts
RUNS_DIR = ROOT_DIR / 'runs'

# Location of compressed datasets on disk
SOURCES_DIR = ROOT_DIR / 'data' / 'sources'


###############################################################################
# Logging parameters
###############################################################################


# Whether to perform benchmarking during evaluation
BENCHMARK = False

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
HIDDEN_CHANNELS = 192

# Hidden dimension channel size
FILTER_CHANNELS = 768

# Convolutional kernel size
KERNEL_SIZE = 3

# (Negative) slope of leaky ReLU activations
LRELU_SLOPE = .1

# The model to use for evaluation.
# One of ['promonet', 'psola', 'world].
MODEL = 'promonet'

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

# Speaker embedding size
SPEAKER_CHANNELS = 256

# Whether to use a two-stage model
TWO_STAGE = False

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
# Training parameters
###############################################################################


# Number of buckets to partition training and validation data into based on
# length to avoid excess padding
BUCKETS = 8

# Set a maximum on the batch size, regardless of frame count
MAX_BATCH_SIZE = 32

# Maximum number of frames in a batch (per GPU)
MAX_FRAMES = 10000

# Number of samples generated during training
CHUNK_SIZE = 8192

# Gradients with norms above this value are clipped to this value
GRADIENT_CLIP_GENERATOR = 1000.

# Number of training steps
NUM_STEPS = 100000

# Number of adaptation steps
NUM_ADAPTATION_STEPS = 5000

# Number of data loading worker threads
NUM_WORKERS = 2

# Seed for all random number generators
RANDOM_SEED = 1234
