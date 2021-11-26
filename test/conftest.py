from pathlib import Path

import pytest

import promovits


###############################################################################
# Testing fixtures
###############################################################################


@pytest.fixture(scope='session')
def audio():
    """Retrieve the test audio"""
    return promovits.load.audio(path('test.wav'))


###############################################################################
# Utilities
###############################################################################


def path(file):
    """Retrieve the path to the test file"""
    return Path(__file__).parent / 'assets' / file
