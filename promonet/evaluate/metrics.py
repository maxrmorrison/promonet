import jiwer
import penn
import ppgs
import torch
import torchutil
import whisper
from whisper.normalizers import EnglishTextNormalizer

import promonet


###############################################################################
# All metrics
###############################################################################


class Metrics:

    def __init__(self):
        self.loudness = torchutil.metrics.RMSE()
        self.periodicity = torchutil.metrics.RMSE()
        self.pitch = Pitch()
        self.ppg = PPG()
        self.wer = WER()
        self.speaker_sim = SpeakerSimilarity()

    def __call__(self):
        result = {
            'loudness': self.loudness(),
            'periodicity': self.periodicity(),
            'pitch': self.pitch(),
            'ppg': self.ppg(),
            'wer': self.wer()}
        if self.speaker_sim.count:
            result['speaker_sim'] = self.speaker_sim()
        return result

    def update(
        self,
        predicted_pitch,
        predicted_periodicity,
        predicted_loudness,
        predicted_ppg,
        target_pitch,
        target_periodicity,
        target_loudness,
        target_ppg,
        wer_args,
        speaker_sim_args=None):
        self.loudness.update(predicted_loudness, target_loudness)
        self.periodicity.update(predicted_periodicity, target_periodicity)
        self.pitch.update(
            predicted_pitch,
            predicted_periodicity,
            target_pitch,
            target_periodicity)
        self.ppg.update(predicted_ppg, target_ppg)
        self.wer.update(*wer_args)
        if speaker_sim_args:
            self.speaker_sim.update(*speaker_sim_args)

    def reset(self):
        self.loudness.reset()
        self.periodicity.reset()
        self.pitch.reset()
        self.ppg.reset()
        self.wer.reset()
        self.speaker_sim.reset()


###############################################################################
# Prosody metrics
###############################################################################


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
                promonet.VOICING_THRESOLD) &
            penn.voicing.threshold(
                target_periodicity,
                promonet.VOICING_THRESOLD))
        predicted = predicted_pitch[voicing]
        target = target_pitch[voicing]

        # Update L1
        super().update(torch.log2(predicted), torch.log2(target))


###############################################################################
# PPG distance metric
###############################################################################


class PPG(torchutil.metrics.Average):
    """PPG distance"""
    def update(self, predicted, target):
        super().update(
            ppgs.distance(
                predicted.squeeze(0),
                target.squeeze(0),
                reduction='sum'),
            predicted.shape[-1])


###############################################################################
# Word error rate metric
###############################################################################


class WER(torchutil.metrics.Average):
    """Word error rate"""
    def update(self, text, audio):
        super().update(
            torch.tensor(
                jiwer.wer(normalize_text(text), speech_to_text(audio))),
            1)


def speech_to_text(audio):
    """Perform speech-to-text using Whisper"""
    # Cache Whisper model
    if (
        not hasattr(speech_to_text, 'model') or
        speech_to_text.device != audio.device
    ):
        model = whisper.load_model('base.en', device=audio.device)
        speech_to_text.model = model
        speech_to_text.device = audio.device

    # Resample audio tensor
    transcribe_input = promonet.resample(
        audio.squeeze(),
        promonet.SAMPLE_RATE,
        16000)

    # Infer text
    text = speech_to_text.model.transcribe(transcribe_input)['text']

    # Normalize
    return normalize_text(text)


def normalize_text(text):
    """Formats text to only words for use in WER"""
    if not hasattr(normalize_text, 'normalizer'):
        normalizer = EnglishTextNormalizer()
        normalize_text.normalizer = normalizer
    return normalize_text.normalizer(text)


###############################################################################
# Speaker similarity metric
###############################################################################


class SpeakerSimilarity(torchutil.metrics.Average):
    """Speaker similarity metric"""
    def update(self, speaker_embed, utterance_embed):
        super().update(torch.abs(speaker_embed - utterance_embed).sum(), 1)
