import string

import pysodic
import torch
from jiwer import wer
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

    def __call__(self):
        return {**self.prosody(), **self.ppg(), **self.wer()}

    def update(self, prosody_args, ppg_args, wer_args):
        self.prosody.update(*prosody_args)
        self.ppg.update(*ppg_args)
        self.wer.update(*wer_args)

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
        predicted_text = predict_text(audio, self.gpu)
        self.total += wer(format_text(gt_text), format_text(predicted_text))
        self.count += 1

    def reset(self):
        self.total = 0.
        self.count = 0

def predict_text(audio, gpu):
    if not hasattr(predict_text, 'model') or predict_text.gpu != gpu:
        device = 'cpu' if gpu is None else f'cuda:{gpu}'
        model = whisper.load_model("base.en", device=device)
        predict_text.model = model
        predict_text.gpu = gpu
    if type(audio) is str: #Provided via file name, let whisper load it itself
        transcribe_input = audio
    else: #Must be tensor
        transcribe_input = promonet.resample(audio, promonet.SAMPLE_RATE, 16000)

    return predict_text.model.transcribe(transcribe_input)['text']

def format_text(text):
    """Formats text to only words for use in WER"""
    if not hasattr(format_text, 'normalizer'):
        normalizer = EnglishTextNormalizer()
        format_text.normalizer = normalizer
    
    return format_text.normalizer(text)