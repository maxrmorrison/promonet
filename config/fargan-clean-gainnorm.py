MODULE = 'promonet'

# Configuration name
CONFIG = 'fargan-clean-gainnorm'

# The model to use.
# One of ['fargan', 'hifigan', 'vocos', 'world'].
MODEL = 'fargan'

# Step to start using adversarial loss
ADVERSARIAL_LOSS_START_STEP = 250000

# Training batch size
BATCH_SIZE = 256

# Training sequence length
CHUNK_SIZE = 4096  # samples

# Whether to use additive noise with FARGAN
FARGAN_ADDITIVE_NOISE = False

# Whether to use gain normalization in the subframe network
FARGAN_GAIN_NORMALIZATION = True

# Whether to use mel spectrogram loss
MEL_LOSS = False

# Whether to use multi-resolution spectral convergence loss
SPECTRAL_CONVERGENCE_LOSS = True
