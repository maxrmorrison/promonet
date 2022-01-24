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
    grid=None,
    target_loudness=None,
    target_pitch=None,
    checkpoint=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform prosody editing"""
    # Maybe use a baseline method instead
    if promovits.MODEL == 'clpcnet':
        return promovits.baseline.clpcnet.from_audio(**locals())
    elif promovits.MODEL == 'fastpitch':
        return promovits.baseline.fastpitch.from_audio(**locals())
    elif promovits.MODEL == 'psola':
        return promovits.baseline.psola.from_audio(**locals())
    elif promovits.MODEL == 'world':
        return promovits.baseline.world.from_audio(**locals())
    elif promovits.MODEL != 'promovits':
        raise ValueError(f'Model {promovits.MODEL} is not recognized')

    # Maybe resample
    with promovits.TIMER('resample'):
        if sample_rate != promovits.SAMPLE_RATE:
            resample_fn = torchaudio.transforms.Resample(
                sample_rate,
                promovits.SAMPLE_RATE)
            audio = resample_fn(audio)

    # Extract prosody features
    with promovits.TIMER('features/prosody'):
        if promovits.PPG_FEATURES:
            pitch, periodicity, loudness, _ = \
                pysodic.from_audio(
                    audio,
                    promovits.SAMPLE_RATE,
                    promovits.HOPSIZE / promovits.SAMPLE_RATE,
                    promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
                    gpu)

            # Maybe interpolate pitch
            if not target_pitch:
                if grid:
                    pitch = promovits.interpolate.pitch(pitch, grid)
                target_pitch = pitch

            # Maybe interpolate periodicity
            if grid:
                periodicity = promovits.interpolate.linear(
                    periodicity,
                    grid)

            # Maybe interpolate loudness
            if not target_loudness:
                if grid:
                    loudness = promovits.interpolate.linear(loudness, grid)
                target_loudness = loudness

    # Get phonetic posteriorgrams
    with promovits.TIMER('features/ppgs'):
        if promovits.PPG_FEATURES:
            features = promovits.preprocess.ppg.from_audio(audio, gpu)

            # Maybe resample length
            if features.shape[1] != audio.shape[1] / promovits.HOPSIZE:
                features = torch.nn.functional.interpolate(
                    features[None],
                    size=audio.shape[1] / promovits.HOPSIZE,
                    mode=promovits.PPG_INTERP_METHOD)[0]

            # Maybe stretch PPGs
            if grid:
                features = promovits.interpolate.ppgs(features, grid)

    # Concatenate features
    with promovits.TIMER('features/concatenate'):
        if promovits.PPG_FEATURES:
            if promovits.LOUDNESS_FEATURES:
                features = torch.cat((features, loudness))
            if promovits.PERIODICITY_FEATURES:
                features = torch.cat((features, periodicity))

    # Convert pitch to indices
    with promovits.TIMER('features/pitch'):
        if target_pitch:
            target_pitch = promovits.convert.hz_to_bins(target_pitch)

    # Maybe get text features
    with promovits.TIMER('features/text'):
        if not promovits.PPG_FEATURES:
            features = promovits.preprocess.text.from_string(text)

    # Setup model
    with promovits.TIMER('load'):
        device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
        if not hasattr(from_audio, 'generator') or \
        from_audio.generator.device != device:
            generator = promovits.model.Generator().to(device)
            generator = promovits.checkpoint.load(
                promovits.checkpoint.latest_path(checkpoint),
                generator)[0]
            generator.eval()
            from_audio.generator = generator

    # Generate audio
    with promovits.TIMER('generate'):
        with torch.no_grad():
            shape = (features.shape[-1],)
            return generator(
                features.to(device)[None],
                torch.tensor(shape, dtype=torch.long, device=device),
                pitch)[0].cpu()


def from_file(
    audio_file,
    target_alignment_file=None,
    target_loudness_file=None,
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
        pitch,
        checkpoint,
        gpu)


def from_file_to_file(
    audio_file,
    output_file,
    target_alignment_file=None,
    target_loudness_file=None,
    target_pitch_file=None,
    checkpoint=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk and save to disk"""
    generated = from_file(
        audio_file,
        target_alignment_file,
        target_loudness_file,
        target_pitch_file,
        checkpoint,
        gpu)
    torchaudio.save(output_file, generated, promovits.SAMPLE_RATE)


def from_files_to_files(
    audio_files,
    output_files,
    target_alignment_files=None,
    target_loudness_files=None,
    target_pitch_files=None,
    checkpoint=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk and save to disk"""
    iterator = zip(
        audio_files,
        output_files,
        target_alignment_files,
        target_loudness_files,
        target_pitch_files)
    for item in iterator:
        from_file_to_file(*item, checkpoint, gpu)
