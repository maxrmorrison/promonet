import time

import promovits


###############################################################################
# Profiling utilities
###############################################################################


class Context:
    """Context manager timer"""

    def __init__(self):
        self.history = {}
        self.start = 0.
        self.name = None

    def __call__(self):
        """Retrieve timer results"""
        if not promovits.BENCHMARK:
            return
        return {
            name: sum(times) / len(times)
            for name, times in self.history.items()}

    def __enter__(self, name):
        """Start the timer"""
        # Don't continue if we aren't benchmarking
        if not promovits.BENCHMARK:
            return

        # Make sure there is no active timer
        if self.name != None:
            raise ValueError('Context timers cannot nest')
        self.name = name

        # Maybe create a timer history for this identifier
        if name not in self.history:
            self.history[name]

        # Start the timer
        self.start = time.time()

    def __exit__(self):
        """Stop the timer"""
        # Don't continue if we aren't benchmarking
        if not promovits.BENCHMARK:
            return

        # Stop the timer
        self.history[self.name].append(time.time() - self.start)

    def reset(self):
        """Reset the timer"""
        self.history = {}
        self.start = 0.
        self.name = None
