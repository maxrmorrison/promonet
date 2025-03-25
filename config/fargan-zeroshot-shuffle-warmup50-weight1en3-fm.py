MODULE = 'promonet'

# Configuration name
CONFIG = 'fargan-zeroshot-shuffle-warmup50-weight1en3-fm'

# The model to use.
# One of ['fargan', 'hifigan', 'vocos', 'world'].
MODEL = 'fargan'

# Step to start using adversarial loss
ADVERSARIAL_LOSS_START_STEP = 300000

# Weight applied to the discriminator loss
ADVERSARIAL_LOSS_WEIGHT = .001

# Training batch size
BATCH_SIZE = 256

# Training sequence length
CHUNK_SIZE = 4096  # samples

# Step to start training discriminator
DISCRIMINATOR_START_STEP = 250000

# Weight applied to the feature matching loss
FEATURE_MATCHING_LOSS_WEIGHT = .001

# Whether to use mel spectrogram loss
MEL_LOSS = False

# Whether to use multi-resolution spectral convergence loss
SPECTRAL_CONVERGENCE_LOSS = True

# Whether to use WavLM x-vectors for zero-shot speaker conditioning
ZERO_SHOT = True

# Whether to shuffle speaker embeddings during training
ZERO_SHOT_SHUFFLE = True
