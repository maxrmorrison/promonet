import math

import jiwer
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
        # self.balance = SpectralBalance()

    def __call__(self):
        result = {
            'pitch': self.pitch(),
            'periodicity': self.periodicity(),
            'ppg': self.ppg()
        } | self.loudness()
        # if self.balance.pitch.count:
        #     result |= self.balance()
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
        # predicted_harmonics=None,
        # target_harmonics=None,
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
        # if predicted_harmonics is not None and target_harmonics is not None:
        #     self.balance.update(
        #         predicted_harmonics,
        #         predicted_periodicity,
        #         predicted_spectrogram,
        #         target_harmonics,
        #         target_periodicity,
        #         target_spectrogram)

    def reset(self):
        # self.balance.reset()
        self.loudness.reset()
        self.periodicity.reset()
        self.pitch.reset()
        self.ppg.reset()
        self.wer.reset()


###############################################################################
# Spectral balance metrics
###############################################################################


class SpectralBalance:

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
            'balance-pitch': self.displacement(),
            'balance-loudness': self.correlation()}

    def update(
        self,
        predicted_harmonics,
        predicted_periodicity,
        predicted_spectrogram,
        target_harmonics,
        target_periodicity,
        target_spectrogram,
        spectral_balance_ratio
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
            iterable = zip(predicted_harmonics, target_harmonics)
        else:
            iterable = zip(predicted_harmonics[1:], target_harmonics[1:])

        # Update metrics
        for f_x, f_y in iterable:
            self.displacement.update(f_x[voicing], f_y[voicing])
        self.correlation.update(
            predicted_centroid[voicing] / target_centroid[voicing],
            spectral_balance_ratio)

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
        if predicted_loudness.ndim == 3:
            predicted_loudness = predicted_loudness.squeeze(0)
        if target_loudness.ndim == 3:
            target_loudness = target_loudness.squeeze(0)

        # Maybe average
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
                    predicted,
                    promonet.SPARSE_PPG_METHOD,
                    promonet.SPARSE_PPG_THRESHOLD)
            target = ppgs.sparsify(
                    target,
                    promonet.SPARSE_PPG_METHOD,
                    promonet.SPARSE_PPG_THRESHOLD)

            # Compute normalized Jensen-Shannon divergence between PPGs
            total = ppgs.distance(
                predicted.squeeze(0),
                target.squeeze(0),
                reduction='sum',
                exponent=self.exponent)

        # Update metric
        super().update(total, predicted.shape[-1])


class WER(torchutil.metrics.Average):
    """Word error rate"""
    def update(self, predicted, target):
        wer = jiwer.wer(target, predicted)
        super().update(torch.tensor(wer), 1)
