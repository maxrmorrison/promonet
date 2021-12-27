import pypar
import pysodic
import torch
import torchaudio

import promovits


###############################################################################
# Promovits generation
###############################################################################


def from_audio(
    audio,
    sample_rate=promovits.SAMPLE_RATE,
    text=None,
    target_alignment=None,
    target_loudness=None,
    target_periodicity=None,
    target_pitch=None,
    checkpoint=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform prosody editing"""
    # TODO - unused arguments

    with promovits.TIMER('resample'):
        # Maybe resample
        if sample_rate != promovits.SAMPLE_RATE:
            resample_fn = torchaudio.transforms.Resample(
                sample_rate,
                promovits.SAMPLE_RATE)
            audio = resample_fn(audio)

    with promovits.TIMER('features/prosody'):
        if promovits.PPG_FEATURES:
            # Get prosody features
            # TODO - Only get these features if needed.
            #        Otherwise, use argument features.
            # TODO - Allow generation without text
            pitch, periodicity, loudness, _ = \
                pysodic.features.from_audio_and_text(
                    audio,
                    promovits.SAMPLE_RATE,
                    text,
                    promovits.HOPSIZE / promovits.SAMPLE_RATE,
                    promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
                    gpu)

    # Get phonetic posteriorgrams
    with promovits.TIMER('features/ppgs'):
        if promovits.PPG_FEATURES:
            features = promovits.preprocess.ppg.from_audio(audio, gpu)

            # Maybe resample length
            if features.shape[1] != pitch.shape[1]:
                features = torch.nn.functional.interpolate(
                    features[None],
                    size=pitch.shape[1],
                    mode=promovits.PPG_INTERP_METHOD)[0]

        # Concatenate features
    with promovits.TIMER('features/concatenate'):
        if promovits.PPG_FEATURES:
            if promovits.LOUDNESS_FEATURES:
                features = torch.cat((features, loudness))
            if promovits.PERIODICITY_FEATURES:
                features = torch.cat((features, periodicity))

    with promovits.TIMER('features/text'):
        if not promovits.PPG_FEATURES:
            # TEMPORARY - text preprocessing is causing deadlock
            # features = promovits.preprocess.text.from_string(text)
            raise NotImplementedError()

    # Setup model
    with promovits.TIMER('load'):
        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
        if not hasattr(from_audio, 'generator') or \
        from_audio.generator.device != device:
            generator = promovits.model.Generator().to(device)
            generator = promovits.load.checkpoint(checkpoint, generator)[0]
            generator.eval()
            from_audio.generator = generator

    # Generate audio
    with promovits.TIMER('generate'):
        with torch.no_grad():
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
    checkpoint=promovits.DEFAULT_CHECKPOINT,
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
        checkpoint,
        gpu)


def from_file_to_file(
    audio_file,
    output_file,
    target_alignment_file=None,
    target_loudness_file=None,
    target_periodicity_file=None,
    target_pitch_file=None,
    checkpoint=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk and save to disk"""
    generated = from_file(
        audio_file,
        target_alignment_file,
        target_loudness_file,
        target_periodicity_file,
        target_pitch_file,
        checkpoint,
        gpu)
    torchaudio.save(output_file, generated, promovits.SAMPLE_RATE)


def from_files_to_files(
    audio_files,
    output_files,
    target_alignment_files=None,
    target_loudness_files=None,
    target_periodicity_files=None,
    target_pitch_files=None,
    checkpoint=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk and save to disk"""
    iterator = zip(
        audio_files,
        output_files,
        target_alignment_files,
        target_loudness_files,
        target_periodicity_files,
        target_pitch_files)
    for item in iterator:
        from_file_to_file(*item, checkpoint, gpu)
