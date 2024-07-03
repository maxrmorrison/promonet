MODULE = 'promonet'

# Configuration name
CONFIG = 'fargan-fdisc'

# The model to use.
# One of ['fargan', 'hifigan', 'vocos', 'world'].
MODEL = 'fargan'

# Step to start using adversarial loss
ADVERSARIAL_LOSS_START_STEP = 300000

# Whether to use the complex multi-band discriminator from RVQGAN
COMPLEX_MULTIBAND_DISCRIMINATOR = False

# Step to start training discriminator
DISCRIMINATOR_START_STEP = 300000

# Training batch size
BATCH_SIZE = 256

# Training sequence length
CHUNK_SIZE = 4096  # samples

# Whether to use the same discriminator as FARGAN
FARGAN_DISCRIMINATOR = True

# Whether to use mel spectrogram loss
MEL_LOSS = False

# Whether to use the multi-period waveform discriminator from HiFi-GAN
MULTI_PERIOD_DISCRIMINATOR = False

# Whether to use multi-resolution spectral convergence loss
SPECTRAL_CONVERGENCE_LOSS = True
