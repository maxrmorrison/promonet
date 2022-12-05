import pysodic
import torch

import promonet


###############################################################################
# All metrics
###############################################################################


class Metrics:

    def __init__(self, gpu):
        self.prosody = pysodic.metrics.Prosody(
            promonet.SAMPLE_RATE,
            promonet.HOPSIZE,
            promonet.WINDOW_SIZE,
            gpu)
        self.ppg = PPG()

    def __call__(self):
        return self.prosody() | self.ppg()

    def update(self, prosody_args, ppg_args):
        self.prosody.update(*prosody_args)
        self.ppg.update(*ppg_args)

    def reset(self):
        self.prosody.reset()
        self.ppg.reset()


###############################################################################
# PPG distance metric
###############################################################################


class PPG:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'ppg': torch.sqrt(self.total / self.count).item()}

    def update(self, predicted, target):
        self.total += ((predicted - target) ** 2).sum()
        self.count += predicted.shape[-1]

    def reset(self):
        self.total = 0.
        self.count = 0
