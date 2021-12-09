import pypar
import torch
import torchaudio

import promovits


###############################################################################
# Promovits inference
###############################################################################


def from_audio(
    hps,
    audio,
    sample_rate=promovits.SAMPLE_RATE,
    text=None,
    target_alignment=None,
    target_pitch=None,
    checkpoint_file=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform prosody editing"""
    # Maybe resample
    if sample_rate != promovits.SAMPLE_RATE:
        resample_fn = torchaudio.transforms.Resample(
            sample_rate,
            promovits.SAMPLE_RATE)
        audio = resample_fn(audio)

    # Get phoneme or PPG features
    if hps.model.use_ppg:
        text = promovits.preprocess.ppg.from_audio(audio, gpu)
    else:
        # TEMPORARY - text preprocessing is causing deadlock
        # text = promovits.preprocess.text.from_string(text)
        raise NotImplementedError()

    # Setup model
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    net_g = promovits.model.Generator(
        len(promovits.preprocess.text.symbols()),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    net_g.eval()

    net_g = promovits.load.checkpoint(checkpoint_file, net_g)[0]

    with torch.no_grad():
        text = text.to(device)
        length = torch.tensor([text.shape[-1]], dtype=torch.long, device=device)

        # TODO - pitch and ppg inputs
        audio = net_g.infer(
            text,
            length,
            noise_scale=.667,
            noise_scale_w=0.8,
            length_scale=1)[0][0,0].cpu()

    return audio

def from_file(
    config,
    audio_file,
    target_alignment_file=None,
    target_pitch_file=None,
    checkpoint_file=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk"""
    # Load audio
    audio = promovits.load.audio(audio_file)

    # Load config
    hps = promovits.load.config(config)

    # Load alignment
    if target_alignment_file:
        alignment = pypar.Alignment(target_alignment_file)
    else:
        alignment = None

    # Load pitch
    if target_pitch_file is None:
        pitch = torch.load(target_pitch_file)
    else:
        pitch = None

    # Generate
    return from_audio(
        hps,
        audio,
        promovits.SAMPLE_RATE,
        alignment,
        pitch,
        checkpoint_file,
        gpu)


def from_file_to_file(
    config,
    audio_file,
    output_file,
    target_alignment_file=None,
    target_pitch_file=None,
    checkpoint_file=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk and save to disk"""
    generated = from_file(
        config,
        audio_file,
        target_alignment_file,
        target_pitch_file,
        checkpoint_file,
        gpu)
    torchaudio.save(output_file, generated, promovits.SAMPLE_RATE)


###############################################################################
# Utilities
###############################################################################


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
