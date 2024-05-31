MODULE = 'promonet'

# Configuration name
CONFIG = 'fargan-loss'

# The model to use.
# One of ['fargan', 'hifigan', 'psola', 'vits', 'vocos', 'world'].
MODEL = 'fargan'

# Step to start using adversarial loss
ADVERSARIAL_LOSS_START_STEP = 75000

# Whether to use mel spectrogram loss
MEL_LOSS = False

# Whether to compare raw audio signals
SIGNAL_LOSS = True

# Whether to use multi-resolution spectral convergence loss
SPECTRAL_CONVERGENCE_LOSS = True
