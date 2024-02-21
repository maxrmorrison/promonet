import math

import jiwer
import numpy as np
import penn
import ppgs
import torch
import torchutil

import promonet


###############################################################################
# All metrics
###############################################################################


class Metrics:

    def __init__(self):
        self.loudness = Loudness()
        self.periodicity = torchutil.metrics.RMSE()
        self.pitch = Pitch()
        self.ppg = PPG()
        self.wer = WER()
        self.speaker_similarity = SpeakerSimilarity()

    def __call__(self):
        result = {
            'loudness': self.loudness(),
            'pitch': self.pitch(),
            'periodicity': self.periodicity(),
            'ppg': self.ppg()}
        if self.speaker_similarity.count:
            result['speaker_similarity'] = self.speaker_similarity()
        if self.wer.count > 0:
            result['wer'] = self.wer()
        return result

    def update(
        self,
        predicted_loudness,
        predicted_pitch,
        predicted_periodicity,
        predicted_ppg,
        target_loudness,
        target_pitch,
        target_periodicity,
        target_ppg,
        predicted_text=None,
        target_text=None,
        speaker_similarity_args=None
    ):
        self.loudness.update(predicted_loudness, target_loudness)
        self.periodicity.update(predicted_periodicity, target_periodicity)
        self.pitch.update(
            predicted_pitch,
            predicted_periodicity,
            target_pitch,
            target_periodicity)
        self.ppg.update(predicted_ppg, target_ppg)
        if predicted_text is not None and target_text is not None:
            self.wer.update(predicted_text, target_text)
        if speaker_similarity_args:
            self.speaker_similarity.update(*speaker_similarity_args)

    def reset(self):
        self.loudness.reset()
        self.periodicity.reset()
        self.pitch.reset()
        self.ppg.reset()
        self.wer.reset()
        self.speaker_similarity.reset()


###############################################################################
# Prosody metrics
###############################################################################


class Loudness(torchutil.metrics.RMSE):
    """Evaluates the average difference in framewise A-weighted loudness"""

    def update(self, predicted_loudness, target_loudness):
        if promonet.LOUDNESS_BANDS > 1:
            predicted_loudness = predicted_loudness.mean(dim=-2, keepdim=True)
            target_loudness = target_loudness.mean(dim=-2, keepdim=True)
        super().update(predicted_loudness, target_loudness)


class Pitch(torchutil.metrics.L1):
    """Evaluates differences in voiced pitch via average error in cents"""

    def __call__(self) -> float:
        """Retrieve the current metric value

        Returns:
            average pitch error in cents
        """
        return 1200 * super().__call__()

    def update(
        self,
        predicted_pitch: torch.Tensor,
        predicted_periodicity: torch.Tensor,
        target_pitch: torch.Tensor,
        target_periodicity: torch.Tensor):
        """Update the metric

        Args:
            predicted_pitch:
                The pitch of the speech being evaluated
                (shape=(1, frames), dtype=torch.float)
            predicted_periodicity:
                The periodicity of the speech being evaluated
                (shape=(1, frames), dtype=torch.long)
            target_pitch:
                The pitch of ground truth speech
                (shape=(1, frames), dtype=torch.float)
            target_periodicity:
                The periodicity of ground truth speech
                (shape=(1, frames), dtype=torch.long)
        """
        # Only evaluate when both predicted and target contain pitch.
        # Otherwise, the magnitude of the error can be arbitrarily large.
        voicing = (
            penn.voicing.threshold(
                predicted_periodicity,
                promonet.VOICING_THRESHOLD) &
            penn.voicing.threshold(
                target_periodicity,
                promonet.VOICING_THRESHOLD))
        predicted = predicted_pitch[voicing]
        target = target_pitch[voicing]

        # Update L1
        super().update(torch.log2(predicted), torch.log2(target))


###############################################################################
# Pronunciation metrics
###############################################################################


class PPG(torchutil.metrics.Average):
    """PPG distance"""

    def __init__(self, exponent=ppgs.SIMILARITY_EXPONENT):
        super().__init__()
        self.exponent = exponent

    def __call__(self):
        result = super().__call__()
        if ppgs.REPRESENTATION_KIND == 'latents':
            return math.sqrt(result)
        return result

    def update(self, predicted, target):
        if ppgs.REPRESENTATION_KIND == 'latents':
            total = (
                (predicted.squeeze(0) - target.squeeze(0)) ** 2).sum(dim=1)
        else:
            total = ppgs.distance(
                predicted.squeeze(0),
                target.squeeze(0),
                reduction='sum',
                exponent=self.exponent)
        super().update(total, predicted.shape[-1])


class WER(torchutil.metrics.Average):
    """Word error rate"""
    def update(self, predicted, target):
        wer = jiwer.wer(target, predicted)
        super().update(torch.tensor(wer), 1)


###############################################################################
# Speaker metrics
###############################################################################


class SpeakerSimilarity(torchutil.metrics.Average):
    """Speaker similarity metric"""
    def update(self, predicted, target):
        super().update(
            torch.nn.functional.cosine_similarity(predicted, target, 0),
            1)
