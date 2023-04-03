MODULE = 'promonet'

# Configuration name
CONFIG = 'spectral-voco-small'

# Whether to use pitch augmentation
AUGMENT_PITCH = True

# Whether to perform gradient clipping on the generator
GRADIENT_CLIP_GENERATOR = 1000.

# Whether to use snake activation in the audio generator
SNAKE = True

# Only use spectral features
SPECTROGRAM_ONLY = True

# Whether to omit latent generation
VOCODER = True
