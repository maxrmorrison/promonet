MODULE = 'promonet'

# Configuration name
CONFIG = 'mels-8layer-sparse3'

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Threshold to sparsify Mel spectrograms
DYNAMIC_RANGE_COMPRESSION_THRESHOLD = 1e-3

# Input features
INPUT_FEATURES = ['spectrogram']

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = False

# Whether to shift Mel inputs to have a minimum of zero
SPARSE_MELS = True

# Whether to shift the Mels given to the Mel loss to have a minimum of zero
SPARSE_MEL_LOSS = True

# Only use spectral features
SPECTROGRAM_ONLY = True

# Number of neural network layers in Vocos
VOCOS_LAYERS = 8
