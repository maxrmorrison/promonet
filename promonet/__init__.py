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
from .config.static import *


###############################################################################
# Module imports
###############################################################################


from .train import train
from . import adapt
from . import baseline
from . import convert
from . import data
from . import edit
from . import evaluate
from . import load
from . import loss
from . import loudness
from . import model
from . import partition
from . import plot
from . import preprocess
from . import speaker
from . import synthesize
