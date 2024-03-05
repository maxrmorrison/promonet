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
        self.formant = Formant()

    def __call__(self):
        result = {
            'loudness': self.loudness(),
            'pitch': self.pitch(),
            'periodicity': self.periodicity(),
            'ppg': self.ppg()}
        if self.formant.l1[0].count:
            result |= self.formant()
        if self.speaker_similarity.count:
            result['speaker_similarity'] = self.speaker_similarity()
        if self.wer.count:
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
        predicted_speaker=None,
        target_speaker=None,
        predicted_formant=None,
        target_formant=None
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
        if predicted_speaker is not None and target_speaker is not None:
            self.speaker_similarity.update(predicted_speaker, target_speaker)
        if predicted_formant is not None and target_formant is not None:
            self.formant.update(
                predicted_formant,
                predicted_periodicity,
                target_formant,
                target_periodicity)

    def reset(self):
        self.formant.reset()
        self.loudness.reset()
        self.periodicity.reset()
        self.pitch.reset()
        self.ppg.reset()
        self.wer.reset()
        self.speaker_similarity.reset()


###############################################################################
# Formant metric
###############################################################################


class Formant:

    def __init__(self, max_formants=promonet.MAX_FORMANTS):
        self.l1 = [torchutil.metrics.L1() for _ in range(max_formants)]

    def __call__(self):
        formants = {f'formant-{i}': self.l1[i]() for i in range(len(self.l1))}
        total = sum(self.l1[i].total for i in range(1, len(self.l1)))
        count = sum(self.l1[i].count for i in range(1, len(self.l1)))
        average = {
            'formant-average': (total / count).item(),
            'pitch-formant-average': (
                (total + self.l1[0].total) / (count + self.l1[0].count)
            ).item()}
        return average | formants

    def update(
        self,
        predicted_formants,
        predicted_periodicity,
        target_formants,
        target_periodicity
    ):
        # Only evaluate when both predicted and target contain pitch.
        # Otherwise, the magnitude of the error can be arbitrarily large.
        voicing = (
            penn.voicing.threshold(
                predicted_periodicity,
                promonet.VOICING_THRESHOLD) &
            penn.voicing.threshold(
                target_periodicity,
                promonet.VOICING_THRESHOLD))

        # Update metric
        for predicted, target, metric in zip(
            predicted_formants,
            target_formants,
            self.l1
        ):
            valid = torch.logical_and(
                voicing,
                torch.logical_and(
                    ~torch.isnan(predicted),
                    ~torch.isnan(target)
                )
            ).squeeze(0)
            metric.update(predicted[valid], target[valid])

    def reset(self):
        for metric in self.l1:
            metric.reset()


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

            # We sparsify the PPGs as to not measure reconstruction of
            # low-probability phoneme bins. Accurate reconstruction of
            # these low-probability bins is typically a sign of noise
            # reconstruction--not pronunciation reconstruction.
            predicted = ppgs.sparsify(
                    predicted.squeeze(0),
                    promonet.SPARSE_PPG_METHOD,
                    promonet.SPARSE_PPG_THRESHOLD)
            target = ppgs.sparsify(
                    target.squeeze(0),
                    promonet.SPARSE_PPG_METHOD,
                    promonet.SPARSE_PPG_THRESHOLD)

            # Compute normalized Jensen-Shannon divergence between PPGs
            total = ppgs.distance(
                predicted,
                target,
                reduction='sum',
                exponent=self.exponent)

        # Update metric
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
