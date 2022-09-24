import pysodic
import torch
import torchaudio

import promonet


###############################################################################
# promonet generation
###############################################################################


def from_audio(
    audio,
    sample_rate=promonet.SAMPLE_RATE,
    text=None,
    grid=None,
    target_loudness=None,
    target_pitch=None,
    checkpoint=promonet.DEFAULT_CHECKPOINT,
    gpu=None):
    """Perform prosody editing"""
    # Maybe use a baseline method instead
    if promonet.MODEL == 'clpcnet':
        with promonet.time.timer('generate'):
            return promonet.baseline.clpcnet.from_audio(**locals())
    elif promonet.MODEL == 'psola':
        with promonet.time.timer('generate'):
            return promonet.baseline.psola.from_audio(**locals())
    elif promonet.MODEL == 'world':
        with promonet.time.timer('generate'):
            return promonet.baseline.world.from_audio(**locals())
    elif promonet.MODEL != 'promonet':
        raise ValueError(f'Model {promonet.MODEL} is not recognized')

    # Maybe resample
    with promonet.time.timer('resample'):
        audio = resample(audio, sample_rate)

    # Extract prosody features
    with promonet.time.timer('features/prosody'):
        if promonet.PPG_FEATURES or promonet.SPECTROGRAM_ONLY:
            pitch, periodicity, loudness, _ = \
                pysodic.from_audio(
                    audio,
                    promonet.SAMPLE_RATE,
                    promonet.HOPSIZE / promonet.SAMPLE_RATE,
                    promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
                    gpu)

            # Maybe interpolate pitch
            if target_pitch is None:
                if grid is not None:
                    pitch = promonet.interpolate.pitch(pitch, grid)
                target_pitch = pitch

            # Maybe interpolate periodicity
            if grid is not None:
                periodicity = promonet.interpolate.grid_sample(
                    periodicity,
                    grid)

            # Maybe interpolate loudness
            if target_loudness is None:
                if grid is not None:
                    loudness = promonet.interpolate.grid_sample(
                        loudness,
                        grid)
                target_loudness = loudness

    # Get phonetic posteriorgrams
    with promonet.time.timer('features/ppgs'):
        if promonet.PPG_FEATURES or promonet.SPECTROGRAM_ONLY:
            features = promonet.preprocess.ppg.from_audio(
                audio,
                sample_rate,
                gpu=gpu)

            # Maybe resample length
            if features.shape[1] != audio.shape[1] // promonet.HOPSIZE:
                align_corners = \
                    None if promonet.PPG_INTERP_METHOD == 'nearest' else False
                features = torch.nn.functional.interpolate(
                    features[None],
                    size=audio.shape[1] // promonet.HOPSIZE,
                    mode=promonet.PPG_INTERP_METHOD,
                    align_corners=align_corners)[0]

            # Maybe stretch PPGs
            if grid is not None:
                features = promonet.interpolate.ppgs(features, grid)

    # Convert pitch to indices
    with promonet.time.timer('features/pitch'):
        if target_pitch is not None:
            target_pitch = promonet.convert.hz_to_bins(target_pitch)

    # Maybe get text features
    with promonet.time.timer('features/text'):
        if not promonet.PPG_FEATURES and not promonet.SPECTROGRAM_ONLY:
            features = promonet.preprocess.text.from_string(text)

    # Setup model
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    if not hasattr(from_audio, 'generator') or from_audio.device != device:
        with promonet.time.timer('load'):
            generator = promonet.model.Generator().to(device)
            if checkpoint.is_dir():
                promonet.checkpoint.latest_path(checkpoint)
            generator = promonet.checkpoint.load(checkpoint, generator)[0]
            generator.eval()
            from_audio.generator = generator
            from_audio.device = device

    # Move features to GPU
    with promonet.time.timer('features/copy'):
        features = features.to(device)[None]
        target_pitch = target_pitch.to(device)
        periodicity = periodicity.to(device)
        target_loudness = target_loudness.to(device)

    # Generate audio
    with promonet.time.timer('generate'):

        # Default length is the entire sequence
        lengths = torch.tensor(
            (features.shape[-1],),
            dtype=torch.long,
            device=device)

        # Default speaker is speaker 0
        speakers = torch.zeros(1, dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            return from_audio.generator(
                features.to(device),
                pitch.to(device),
                periodicity.to(device),
                loudness.to(device),
                lengths,
                speakers)[0][0].cpu()


def from_file(
    audio_file,
    text_file=None,
    grid_file=None,
    target_loudness_file=None,
    target_pitch_file=None,
    checkpoint=promonet.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk"""
    # Load audio
    audio = promonet.load.audio(audio_file)

    # Load text
    if text_file is None:
        text = None
    else:
        text = promonet.load.text(text_file)

    # Load alignment
    if grid_file is None:
        grid = None
    else:
        grid = torch.load(grid_file)

    # Load loudness
    if target_loudness_file is None:
        loudness = None
    else:
        loudness = torch.load(target_loudness_file)

    # Load pitch
    if target_pitch_file is None:
        pitch = None
    else:
        pitch = torch.load(target_pitch_file)

    # Generate
    return from_audio(
        audio,
        promonet.SAMPLE_RATE,
        text,
        grid,
        loudness,
        pitch,
        checkpoint,
        gpu)


def from_file_to_file(
    audio_file,
    output_file,
    text_file=None,
    grid_file=None,
    target_loudness_file=None,
    target_pitch_file=None,
    checkpoint=promonet.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk and save to disk"""
    generated = from_file(
        audio_file,
        text_file,
        grid_file,
        target_loudness_file,
        target_pitch_file,
        checkpoint,
        gpu)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    torchaudio.save(output_file, generated, promonet.SAMPLE_RATE)


def from_files_to_files(
    audio_files,
    output_files,
    text_files=None,
    grid_files=None,
    target_loudness_files=None,
    target_pitch_files=None,
    checkpoint=promonet.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk and save to disk"""
    # Handle None arguments
    if grid_files is None:
        grid_files = [None] * len(audio_files)
    if text_files is None:
        text_files = [None] * len(audio_files)
    if target_loudness_files is None:
        target_loudness_files = [None] * len(audio_files)
    if target_pitch_files is None:
        target_pitch_files = [None] * len(audio_files)

    # Perform prosody editing
    iterator = zip(
        audio_files,
        output_files,
        text_files,
        grid_files,
        target_loudness_files,
        target_pitch_files)
    for item in iterator:
        from_file_to_file(*item, checkpoint=checkpoint, gpu=gpu)


###############################################################################
# Utilities
###############################################################################


def resample(audio, sample_rate, target_rate=promonet.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)
