# TODO - world pitch reconstruction
# TODO - analyze interpolation of loudness
# TODO - consider PPG reinforcement in the discrim if pronunciation is bad
# TODO - address loudness-scaling acting as bias in V/UV F1 computation (by rescaling)
# TODO - fix all warnings
# TODO - fix shape error during adaptation
# TODO - ppg duration metric
# TODO - loss balancing


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
