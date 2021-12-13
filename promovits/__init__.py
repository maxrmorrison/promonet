###############################################################################
# Configuration
###############################################################################


import argparse
import importlib.util
from pathlib import Path

from .constants import *


def configure():
    """Perform configuration"""
    # Get config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIGURATION,
        help='The configuration file')
    config = parser.parse_args().config

    # Load config file as a module
    config_spec = importlib.util.spec_from_file_location('config', config)
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)

    # Merge config module and default config module
    for parameter in dir(config_module):
        if not hasattr(promovits, parameter):
            raise ValueError(
                f'Configuraton parameter {parameter} is not defined')
        setattr(promovits, parameter, getattr(config_module, parameter))


# Perform configuration before module imports
configure()


###############################################################################
# Module imports
###############################################################################


from .core import *
from . import data
from . import evaluate
from . import load
from . import loss
from . import model
from . import preprocess
from . import write
