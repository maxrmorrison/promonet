import pypar
import pysodic
import torch
import torchaudio

import promovits


###############################################################################
# Promovits inference
###############################################################################


def from_audio(
    audio,
    sample_rate=promovits.SAMPLE_RATE,
    text=None,
    target_alignment=None,
    target_loudness=None,
    target_periodicity=None,
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

    if promovits.PPG_FEATURES:

        # Get prosody features
        # TODO - Only get these features if needed.
        #        Otherwise, use argument features.
        # TODO - Allow generation without text
        pitch, periodicity, loudness, _ = pysodic.features.from_audio_and_text(
            audio,
            promovits.SAMPLE_RATE,
            text,
            promovits.HOPSIZE / promovits.SAMPLE_RATE,
            promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
            gpu)

        # Get phonetic posteriorgrams
        features = promovits.preprocess.ppg.from_audio(audio, gpu)

        # Maybe resample length
        if features.shape[1] != pitch.shape[1]:
            features = torch.nn.functional.interpolate(
                features[None],
                size=pitch.shape[1],
                mode=promovits.PPG_INTERP_METHOD)[0]

        # Concatenate features
        if promovits.LOUDNESS_FEATURES:
            features = torch.cat((features, loudness))
        if promovits.PERIODICITY_FEATURES:
            features = torch.cat((features, periodicity))

    else:
        # TEMPORARY - text preprocessing is causing deadlock
        # features = promovits.preprocess.text.from_string(text)
        raise NotImplementedError()

    # Setup model
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    generator = promovits.model.Generator().to(device)
    generator = promovits.load.checkpoint(checkpoint_file, generator)[0]
    generator.eval()

    with torch.no_grad():

        # Generate audio
        shape = (features.shape[-1],)
        return generator(
            features.to(device),
            torch.tensor(shape, dtype=torch.long, device=device),
            pitch)[0][0].cpu()


def from_file(
    audio_file,
    target_alignment_file=None,
    target_loudness_file=None,
    target_periodicity_file=None,
    target_pitch_file=None,
    checkpoint_file=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk"""
    # Load audio
    audio = promovits.load.audio(audio_file)

    # Load alignment
    if target_alignment_file:
        alignment = pypar.Alignment(target_alignment_file)
    else:
        alignment = None

    # Load loudness
    if target_loudness_file:
        loudness = torch.load(target_loudness_file)
    else:
        loudness = None

    # Load periodicity
    if target_periodicity_file:
        periodicity = torch.load(target_periodicity_file)
    else:
        periodicity = None

    # Load pitch
    if target_pitch_file is None:
        pitch = torch.load(target_pitch_file)
    else:
        pitch = None

    # Generate
    return from_audio(
        audio,
        promovits.SAMPLE_RATE,
        alignment,
        loudness,
        periodicity,
        pitch,
        checkpoint_file,
        gpu)


def from_file_to_file(
    audio_file,
    output_file,
    target_alignment_file=None,
    target_loudness_file=None,
    target_periodicity_file=None,
    target_pitch_file=None,
    checkpoint_file=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk and save to disk"""
    generated = from_file(
        audio_file,
        target_alignment_file,
        target_loudness_file,
        target_periodicity_file,
        target_pitch_file,
        checkpoint_file,
        gpu)
    torchaudio.save(output_file, generated, promovits.SAMPLE_RATE)
