###############################################################################
# Configuration
###############################################################################


import importlib.util
import sys
from pathlib import Path

# Default configuration parameters to be modified
from . import defaults


def configure():
    """Perform configuration"""
    # Get config file
    try:
        index = sys.argv.index('--config')
    except:
        return
    if index == -1 or index + 1 == len(sys.argv):
        return
    config = Path(sys.argv[index + 1])

    # Load config file as a module
    config_spec = importlib.util.spec_from_file_location('config', config)
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)

    # Merge config module and default config module
    for parameter in dir(config_module):
        if not hasattr(defaults, parameter):
            raise ValueError(
                f'Configuraton parameter {parameter} is not defined')
        setattr(defaults, parameter, getattr(config_module, parameter))


# Perform configuration before module imports
configure()

# Import configuration parameters
from .defaults import *
from .static import *


###############################################################################
# Module imports
###############################################################################


from .core import *
from . import convert
from . import data
from . import evaluate
from . import load
from . import loss
from . import model
from . import preprocess
from . import write
