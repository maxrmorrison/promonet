MODULE = 'promonet'

# Configuration name
CONFIG = 'fargan'

# The model to use.
# One of ['fargan', 'hifigan', 'psola', 'vocos', 'world'].
MODEL = 'fargan'

# Step to start using adversarial loss
ADVERSARIAL_LOSS_START_STEP = 250000

# Training batch size
BATCH_SIZE = 256

# Training sequence size in samples
CHUNK_SIZE = 4096

# Whether to use mel spectrogram loss
MEL_LOSS = False

# Whether to use multi-resolution spectral convergence loss
SPECTRAL_CONVERGENCE_LOSS = True
