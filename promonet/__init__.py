# Development
# - debug ppgs**
# - clean-up*
# - xFormers
# - subjective evals
# - Am I clipping pitch above fmax or below fmin? Error analysis
# - HN-USFGAN baseline


###############################################################################
# Configuration
###############################################################################


# Default configuration parameters to be modified
from .config import defaults

# Modify configuration
import yapecs
yapecs.configure('promonet', defaults)

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
from . import train
from . import write
