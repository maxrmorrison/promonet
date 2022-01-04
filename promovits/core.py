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
    stretch=None,
    target_loudness=None,
    target_periodicity=None,
    target_pitch=None,
    checkpoint=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform prosody editing"""
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
            extract_prosody = (
                (promovits.PITCH_FEATURES and target_pitch is None) or
                (promovits.PERIODICITY_FEATURES and target_periodicity is None)
                or (promovits.LOUDNESS_FEATURES and target_loudness is None))
            if extract_prosody:
                pitch, periodicity, loudness, _ = \
                    pysodic.from_audio(
                        audio,
                        promovits.SAMPLE_RATE,
                        promovits.HOPSIZE / promovits.SAMPLE_RATE,
                        promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
                        gpu)

                # Maybe interpolate pitch
                if not target_pitch:
                    if stretch:
                        pitch = promovits.interpolate.pitch(pitch, stretch)
                    target_pitch = pitch

                # Maybe interpolate periodicity
                if not target_periodicity:
                    if stretch:
                        periodicity = promovits.interpolate.linear(
                            periodicity,
                            stretch)
                    target_periodicity = periodicity

                # Maybe interpolate loudness
                if not target_loudness:
                    if stretch:
                        loudness = promovits.interpolate.linear(
                            loudness,
                            stretch)
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
            if stretch:
                features = promovits.interpolate.ppgs(features, stretch)

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
            # TEMPORARY - text preprocessing is causing deadlock
            # features = promovits.preprocess.text.from_string(text)
            raise NotImplementedError()

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
