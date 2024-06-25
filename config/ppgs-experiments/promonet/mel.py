MODULE = 'promonet'

# Configuration name
CONFIG = 'mel'

# Whether to use loudness augmentation
AUGMENT_LOUDNESS = False

# Whether to use pitch augmentation
AUGMENT_PITCH = False

# Batch size
BATCH_SIZE = 32

# Number of samples generated during training
CHUNK_SIZE = 8192

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = False

# Evaluation ratios for pitch-shifting, time-stretching, and loudness-scaling
EVALUATION_RATIOS = [.891, 1.12]

# Input features
INPUT_FEATURES = ['pitch', 'ppg']

# The model to use. One of ['hifigan', 'psola', 'vits', 'vocos', 'world'].
MODEL = 'vits'

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = True

# Type of sparsification used for ppgs
# One of ['constant', 'percentile', 'topk', None]
SPARSE_PPG_METHOD = None

# Number of training steps
# STEPS = 250000
STEP = 800000

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = False

# Whether to perform Viterbi decoding on pitch features
VITERBI_DECODE_PITCH = False
