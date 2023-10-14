import jiwer
import ppgs
import pysodic
import torch
import numpy as np
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
        self.wer = WER(gpu)
        self.speaker_sim = SpeakerSimilarity(gpu)

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

    def __init__(self, gpu):
        self.reset()
        self.gpu = gpu

    def __call__(self):
        return {'wer': self.total / self.count}

    def update(self, gt_text, audio):
        predicted_text = speech_to_text(audio, self.gpu)
        self.total += jiwer.wer(normalize_text(gt_text), predicted_text)
        self.count += 1

    def reset(self):
        self.total = 0.
        self.count = 0


def speech_to_text(audio, gpu):
    """Perform speech-to-text using Whisper"""
    # Cache Whisper model
    if not hasattr(speech_to_text, 'model') or speech_to_text.gpu != gpu:
        device = 'cpu' if gpu is None else f'cuda:{gpu}'
        model = whisper.load_model('base.en', device=device)
        speech_to_text.model = model
        speech_to_text.gpu = gpu

    # Get audio
    if isinstance(audio, torch.Tensor):

        # Resample audio tensor
        transcribe_input = promonet.resample(
            audio,
            promonet.SAMPLE_RATE,
            16000)

    else:

        # Assume audio is a filename
        transcribe_input = audio

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

    def __init__(self, gpu):
        self.reset()

    def __call__(self):
        if self.count == 0:
            return {}
        return {'speaker_sim': self.total / self.count}

    def update(self, speaker_embed, utterance_embed):
        diff = np.sum(np.abs(speaker_embed - utterance_embed))
        self.total += diff
        self.count += 1

    def reset(self):
        self.total = 0.
        self.count = 0
