MODULE = 'promonet'

# Configuration name
CONFIG = 'fargan-long-noadv'

# The model to use.
# One of ['fargan', 'hifigan', 'vocos', 'world'].
MODEL = 'fargan'

# Step to start using adversarial loss
ADVERSARIAL_LOSS_START_STEP = 1000000

# Training batch size
BATCH_SIZE = 1024

# Training sequence length
CHUNK_SIZE = 4096  # samples

# Whether to use mel spectrogram loss
MEL_LOSS = False

# Whether to use multi-resolution spectral convergence loss
SPECTRAL_CONVERGENCE_LOSS = True
