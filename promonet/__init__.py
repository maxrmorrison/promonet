# Trainings
# - cosine templates
# - longer two-stage
# - two-stage without MRD
# - scale-up promonet
# - scale-up promovoco
# - scale-up spectral
# - scale-up spectral + latent
# - method ablations

# Development
# - xFormers **
# - new ppgs
# - subjective evals
# - Am I clipping pitch above fmax or below fmin? Error analysis
# - HN-USFGAN baseline

# Low priority
# - address loudness-scaling acting as bias in V/UV F1 computation (by rescaling)


###############################################################################
# Configuration
###############################################################################


# Default configuration parameters to be modified
from .config import defaults

# Modify configuration
import yapecs
yapecs.configure(defaults)

# Import configuration parameters
from .config.defaults import *
from . import time
from .config.static import *


###############################################################################
# Module imports
###############################################################################


from .core import *
from . import baseline
from . import checkpoint
from . import convert
from . import data
from . import evaluate
from . import interpolate
from . import load
from . import loss
from . import loudness
from . import model
from . import partition
from . import plot
from . import preprocess
from . import train
from . import write
