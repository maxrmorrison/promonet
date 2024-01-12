MODULE = 'promonet'

# Configuration name
CONFIG = 'mels-8layer'

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = True

# Input features
INPUT_FEATURES = ['spectrogram']

# Whether to use the multi-scale waveform discriminator from MelGAN
MULTI_SCALE_DISCRIMINATOR = False

# Only use spectral features
SPECTROGRAM_ONLY = True

# Number of neural network layers in Vocos
VOCOS_LAYERS = 8
