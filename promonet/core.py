import contextlib
import os

import pyfoal
import pysodic
import torch
import torchaudio
import tqdm

import promonet


###############################################################################
# Generation API
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
    if promonet.MODEL == 'psola':
        with promonet.time.timer('generate'):
            return promonet.baseline.psola.from_audio(**locals())
    elif promonet.MODEL == 'world':
        with promonet.time.timer('generate'):
            return promonet.baseline.world.from_audio(**locals())
    elif promonet.MODEL != 'promonet':
        raise ValueError(f'Model {promonet.MODEL} is not recognized')

    # Preprocess
    features, target_pitch, periodicity, target_loudness = preprocess(
        audio,
        sample_rate,
        text,
        grid,
        target_loudness,
        target_pitch,
        gpu)

    # Generate
    return generate(
        features,
        target_pitch,
        periodicity,
        target_loudness,
        checkpoint)


def from_file(
    audio_file,
    text_file=None,
    grid_file=None,
    target_loudness_file=None,
    target_pitch_file=None,
    checkpoint=promonet.DEFAULT_CHECKPOINT,
    gpu=None):
    """Edit speech on disk"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

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
        grid = torch.load(grid_file).to(device)

    # Load loudness
    if target_loudness_file is None:
        loudness = None
    else:
        loudness = torch.load(target_loudness_file, map_location=device)

    # Load pitch
    if target_pitch_file is None:
        pitch = None
    else:
        pitch = torch.load(target_pitch_file, map_location=device)

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
        gpu).to(device='cpu', dtype=torch.float32)
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
# Generation pipeline
###############################################################################


def generate(
    features,
    pitch,
    periodicity,
    loudness,
    checkpoint=promonet.DEFAULT_CHECKPOINT):
    """Generate speech from phoneme and prosody features"""
    device = features.device

    with promonet.time.timer('load'):

        # Cache model
        if not hasattr(generate, 'model') or generate.device != device:
            model = promonet.model.Generator().to(device)
            if checkpoint.is_dir():
                checkpoint = promonet.checkpoint.latest_path(checkpoint)
            model = promonet.checkpoint.load(checkpoint, model)[0]
            generate.model = model
            generate.device = device

    with promonet.time.timer('generate'):

        # Default length is the entire sequence
        lengths = torch.tensor(
            (features.shape[-1],),
            dtype=torch.long,
            device=device)

        # Default speaker is speaker 0
        speakers = torch.zeros(1, dtype=torch.long, device=device)

        # Generate
        with generation_context(generate.model):
            return generate.model(
                features,
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers)[0][0].cpu()


def preprocess(
    audio,
    sample_rate=promonet.SAMPLE_RATE,
    text=None,
    grid=None,
    target_loudness=None,
    target_pitch=None,
    gpu=None):
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

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
                    gpu=gpu)

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

        # Convert pitch to indices
        if target_pitch is not None:
            target_pitch = promonet.convert.hz_to_bins(target_pitch)

    with promonet.time.timer('features/phonemes'):

        # Grapheme-to-phoneme
        if not promonet.PPG_FEATURES and not promonet.SPECTROGRAM_ONLY:
            _, features = pyfoal.g2p.from_text(text, to_indices=True)

        # Phonetic posteriorgrams
        if promonet.PPG_FEATURES or promonet.SPECTROGRAM_ONLY:
            features = promonet.data.preprocess.ppg.from_audio(
                audio,
                sample_rate,
                gpu=gpu)

            # Maybe resample length
            frames = promonet.convert.samples_to_frames(audio.shape[1])
            if features.shape[1] != frames:
                align_corners = \
                    None if promonet.PPG_INTERP_METHOD == 'nearest' else False
                features = torch.nn.functional.interpolate(
                    features[None],
                    size=frames,
                    mode=promonet.PPG_INTERP_METHOD,
                    align_corners=align_corners)[0]

            # Maybe stretch PPGs
            if grid is not None:
                features = promonet.interpolate.ppgs(features, grid)

        features = features.to(device)[None]

    return features, target_pitch, periodicity, target_loudness


###############################################################################
# Utilities
###############################################################################


@contextlib.contextmanager
def chdir(directory):
    """Context manager for changing the current working directory"""
    curr_dir = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(curr_dir)


@contextlib.contextmanager
def generation_context(model):
    device_type = next(model.parameters()).device.type

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.no_grad():

        # Automatic mixed precision on GPU
        if device_type == 'cuda':
            with torch.autocast(device_type):
                yield

        else:
            yield

    # Prepare model for training
    model.train()


def iterator(iterable, message, initial=0, total=None):
    """Create a tqdm iterator"""
    total = len(iterable) if total is None else total
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        initial=initial,
        total=total)


def resample(audio, sample_rate, target_rate=promonet.SAMPLE_RATE):
    """Perform audio resampling"""
    if sample_rate == target_rate:
        return audio
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    resampler = resampler.to(audio.device)
    return resampler(audio)
