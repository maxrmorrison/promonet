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
        # self.formant = Formant()

    def __call__(self):
        result = {
            'pitch': self.pitch(),
            'periodicity': self.periodicity(),
            'ppg': self.ppg()
        } | self.loudness()
        # if self.formant.pitch.count:
        #     result |= self.formant()
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
        # predicted_formant=None,
        # target_formant=None,
        # predicted_spectrogram=None,
        # target_spectrogram=None,
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
        # if predicted_formant is not None and target_formant is not None:
        #     self.formant.update(
        #         predicted_formant,
        #         predicted_periodicity,
        #         predicted_spectrogram,
        #         target_formant,
        #         target_periodicity,
        #         target_spectrogram)

    def reset(self):
        # self.formant.reset()
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

    def __init__(
        self,
        predicted_stats,
        target_stats,
        include_fundamental=False
    ):
        self.include_fundamental = include_fundamental
        self.displacement = torchutil.metrics.L1()
        self.correlation = torchutil.metrics.PearsonCorrelation(
            *predicted_stats(),
            *target_stats())

    def __call__(self):
        return {
            'formant-pitch': self.displacement(),
            'formant-loudness': self.correlation()}

    def update(
        self,
        predicted_formants,
        predicted_periodicity,
        predicted_spectrogram,
        target_formants,
        target_periodicity,
        target_spectrogram,
        formant_ratio
    ):
        # Only evaluate when both predicted and target contain pitch.
        # Otherwise, the magnitude of the error can be arbitrarily large.
        voicing = (
            penn.voicing.threshold(
                predicted_periodicity,
                promonet.VOICING_THRESHOLD) &
            penn.voicing.threshold(
                target_periodicity,
                promonet.VOICING_THRESHOLD)
        ).squeeze(0)

        # Get framewise spectral centroid
        predicted_centroid = spectral_centroid(predicted_spectrogram)
        target_centroid = spectral_centroid(target_spectrogram)

        # Maybe include fundamental
        if self.include_fundamental:
            iterable = zip(predicted_formants, target_formants)
        else:
            iterable = zip(predicted_formants[1:], target_formants[1:])

        # Update metrics
        for f_x, f_y in iterable:
            self.displacement.update(f_x[voicing], f_y[voicing])
        self.correlation.update(
            predicted_centroid[voicing] / target_centroid[voicing],
            formant_ratio)

    def reset(self):
        self.displacement.reset()
        self.correlation.reset()


def spectral_centroid(spectrogram):
    # Compute STFT frequencies
    frequencies = torch.abs(torch.fft.fftfreq(
        2 * (spectrogram.shape[0] - 1),
        1 / promonet.SAMPLE_RATE,
        device=spectrogram.device
    )[:spectrogram.shape[0]])

    # Compute centroid
    return (
        frequencies * spectrogram.T
    ).sum(dim=1) / spectrogram.sum(dim=0).squeeze()


###############################################################################
# Prosody metrics
###############################################################################


class Loudness:
    """Evaluates the average difference in framewise A-weighted loudness"""

    def __init__(self, threshold=-60.):
        self.threshold = threshold
        self.loud = torchutil.metrics.RMSE()
        self.quiet = torchutil.metrics.RMSE()
        self.both = torchutil.metrics.RMSE()

    def __call__(self):
        return {
            'loudness': self.both(),
            'loudness-loud': self.loud(),
            'loudness-quiet': self.quiet()}

    def update(self, predicted_loudness, target_loudness):
        # Maybe average
        if promonet.LOUDNESS_BANDS > 1:
            predicted_loudness = predicted_loudness.mean(dim=-2, keepdim=True)
            target_loudness = target_loudness.mean(dim=-2, keepdim=True)

        # Update
        loud = torch.logical_and(
            predicted_loudness > self.threshold,
            target_loudness > self.threshold)
        self.loud.update(predicted_loudness[loud], target_loudness[loud])
        self.quiet.update(predicted_loudness[~loud], target_loudness[~loud])
        self.both.update(predicted_loudness, target_loudness)

    def reset(self):
        self.loud.reset()
        self.quiet.reset()
        self.both.reset()


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
