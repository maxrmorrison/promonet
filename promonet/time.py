import contextlib
import time

import promonet


###############################################################################
# Profiling utilities
###############################################################################


class Context:
    """Context manager timer"""

    def __init__(self):
        self.reset()

    def __call__(self):
        """Retrieve timer results"""
        return {name: sum(times) for name, times in self.history.items()}

    def __enter__(self):
        """Start the timer"""
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer"""
        elapsed = time.time() - self.start

        # Add to timer history
        if self.name not in self.history:
            self.history[self.name] = [elapsed]
        else:
            self.history[self.name].append(elapsed)

    def reset(self):
        """Reset the timer"""
        self.history = {}
        self.start = 0.
        self.name = None


@contextlib.contextmanager
def timer(name):
    """Wrapper to handle context changes of global timer"""
    # Don't continue if we aren't benchmarking
    if not promonet.BENCHMARK:
        yield
        return

    promonet.TIMER.name = name
    with promonet.TIMER:
        yield
    promonet.TIMER.name = None
