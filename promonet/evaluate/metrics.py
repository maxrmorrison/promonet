import jiwer
import ppgs
import pysodic
import torch
import whisper
from whisper.normalizers import EnglishTextNormalizer

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
        self.wer = WER()
        self.speaker_sim = SpeakerSimilarity()

    def __call__(self):
        return {
            **self.prosody(),
            **self.ppg(),
            **self.wer(),
            **self.speaker_sim()}

    def update(self, prosody_args, ppg_args, wer_args, speaker_sim_args=None):
        self.prosody.update(*prosody_args)
        self.ppg.update(*ppg_args)
        self.wer.update(*wer_args)
        if speaker_sim_args:
            self.speaker_sim.update(*speaker_sim_args)

    def reset(self):
        self.prosody.reset()
        self.ppg.reset()
        self.wer.reset()
        self.speaker_sim.reset()


###############################################################################
# PPG distance metric
###############################################################################


class PPG:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'ppg': torch.sqrt(self.total / self.count).item()}

    def update(self, predicted, target):
        for pred, targ in zip(predicted, target):
            self.total += ppgs.distance(pred.T, targ.T, reduction='sum')
            self.count += pred.shape[-1]

    def reset(self):
        self.total = 0.
        self.count = 0


###############################################################################
# Word error rate metric
###############################################################################


class WER:

    def __init__(self):
        self.reset()

    def __call__(self):
        return {'wer': self.total / self.count}

    def update(self, text, audio):
        predicted_text = speech_to_text(audio)
        self.total += jiwer.wer(normalize_text(text), predicted_text)
        self.count += 1

    def reset(self):
        self.total = 0.
        self.count = 0


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


class SpeakerSimilarity:

    def __init__(self):
        self.reset()

    def __call__(self):
        if self.count == 0:
            return {}
        return {'speaker_sim': (self.total / self.count).item()}

    def update(self, speaker_embed, utterance_embed):
        self.total += torch.abs(speaker_embed - utterance_embed).sum()
        self.count += 1

    def reset(self):
        self.total = 0.
        self.count = 0
