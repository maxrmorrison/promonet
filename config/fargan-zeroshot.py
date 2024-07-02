MODULE = 'promonet'

# Configuration name
CONFIG = 'fargan-zeroshot'

# The model to use.
# One of ['fargan', 'hifigan', 'vocos', 'world'].
MODEL = 'fargan'

# Step to start using adversarial loss
ADVERSARIAL_LOSS_START_STEP = 250000

# Training batch size
BATCH_SIZE = 256

# Training sequence length
CHUNK_SIZE = 4096  # samples

# Whether to use mel spectrogram loss
MEL_LOSS = False

# Whether to use multi-resolution spectral convergence loss
SPECTRAL_CONVERGENCE_LOSS = True

# Whether to use WavLM x-vectors for zero-shot speaker conditioning
ZERO_SHOT = True
