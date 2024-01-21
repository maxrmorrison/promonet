MODULE = 'promonet'

# Configuration name
CONFIG = 'augment-multiband-varpitch-256-constant005-hifigan-viterbi-loudness-sparseloss1e4'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Whether to use loudness augmentation
AUGMENT_LOUDNESS = True

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Threshold to sparsify Mel spectrograms
DYNAMIC_RANGE_COMPRESSION_THRESHOLD = 1e-4

# The model to use. One of ['hifigan', 'psola', 'vits', 'vocos', 'world'].
MODEL = 'hifigan'

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = False

# Whether to shift the Mels given to the Mel loss to have a minimum of zero
SPARSE_MEL_LOSS = True

# Type of sparsification used for ppgs
# One of ['constant', 'percentile', 'topk', None]
SPARSE_PPG_METHOD = 'constant'

# Threshold for ppg sparsification.
# In [0, 1] for 'contant' and 'percentile'; integer > 0 for 'topk'.
SPARSE_PPG_THRESHOLD = 0.05

# Whether to use variable-width pitch bins
VARIABLE_PITCH_BINS = True

# Whether to perform Viterbi decoding on pitch features
VITERBI_DECODE_PITCH = True
