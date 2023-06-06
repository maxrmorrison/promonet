import contextlib
import os
from typing import List, Optional, Union

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
    audio: torch.Tensor,
    sample_rate: int = promonet.SAMPLE_RATE,
    text: Optional[str] = None,
    grid: Optional[torch.Tensor] = None,
    target_loudness: Optional[torch.Tensor] = None,
    target_pitch: Optional[torch.Tensor] = None,
    checkpoint: Union[str, os.PathLike]=promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None) -> torch.Tensor:
    """Perform speech editing

    Args:
        audio: The audio to edit
        sample_rate: The audio sample rate
        text: The speech transcript for editing phoneme durations
        grid: The interpolation grid for editing phoneme durations
        target_loudness: The loudness contour for editing loudness
        target_pitch: The pitch contour for shifting pitch
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        edited: The edited audio
    """
    # Maybe use a baseline method instead
    if promonet.MODEL == 'psola':
        with promonet.time.timer('generate'):
            return promonet.baseline.psola.from_audio(**locals())
    elif promonet.MODEL == 'world':
        with promonet.time.timer('generate'):
            return promonet.baseline.world.from_audio(**locals())

    # Preprocess
    with promonet.time.timer('preprocess'):
        (
            features,
            target_pitch,
            periodicity,
            target_loudness,
            spectrograms
        ) = preprocess(
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
        spectrograms,
        checkpoint)


def from_file(
    audio_file: Union[str, os.PathLike],
    text_file: Optional[Union[str, os.PathLike]] = None,
    grid_file: Optional[Union[str, os.PathLike]] = None,
    target_loudness_file: Optional[Union[str, os.PathLike]] = None,
    target_pitch_file: Optional[Union[str, os.PathLike]] = None,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None) -> torch.Tensor:
    """Edit speech on disk

    Args:
        audio_file: The audio to edit
        text_file: The speech transcript for editing phoneme durations
        grid_file: The interpolation grid for editing phoneme durations
        target_loudness_file: The loudness contour for editing loudness
        target_pitch_file: The pitch contour for shifting pitch
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        edited: The edited audio
    """
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
    audio_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    text_file: Optional[Union[str, os.PathLike]] = None,
    grid_file: Optional[Union[str, os.PathLike]] = None,
    target_loudness_file: Optional[Union[str, os.PathLike]] = None,
    target_pitch_file: Optional[Union[str, os.PathLike]] = None,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None):
    """Edit speech on disk and save to disk

    Args:
        audio_file: The audio to edit
        output_file: The file to save the edited audio
        text_file: The speech transcript for editing phoneme durations
        grid_file: The interpolation grid for editing phoneme durations
        target_loudness_file: The loudness contour for editing loudness
        target_pitch_file: The pitch contour for shifting pitch
        checkpoint: The generator checkpoint
        gpu: The GPU index
    """
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
    audio_files: List[Union[str, os.PathLike]],
    output_files: List[Union[str, os.PathLike]],
    text_files: Optional[List[Union[str, os.PathLike]]] = None,
    grid_files: Optional[List[Union[str, os.PathLike]]] = None,
    target_loudness_files: Optional[List[Union[str, os.PathLike]]] = None,
    target_pitch_files: Optional[List[Union[str, os.PathLike]]] = None,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None):
    """Edit speech on disk and save to disk

    Args:
        audio_files: The audio to edit
        output_files: The files to save the edited audio
        text_files: The speech transcripts for editing phoneme durations
        grid_files: The interpolation grids for editing phoneme durations
        target_loudness_files: The loudness contours for editing loudness
        target_pitch_files: The pitch contours for shifting pitch
        checkpoint: The generator checkpoint
        gpu: The GPU index
    """
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
    spectrograms,
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
                speakers,
                spectrograms=spectrograms)[0][0].cpu()


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
    audio = resample(audio, sample_rate)

    if promonet.MODEL == 'vits':

        # Grapheme-to-phoneme
        _, features = pyfoal.g2p.from_text(text, to_indices=True)
        target_pitch, periodicity, target_loudness = None, None, None

    else:

        # Extract prosody features
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

        # Phonetic posteriorgrams
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

    if promonet.MODEL == 'hifigan':

        # Compute spectrogram
        spectrograms = promonet.data.preprocess.spectrogram.from_audio(
            audio
        )[None].to(device)

        # Maybe stretch spectrogram
        if grid is not None:
            spectrograms = promonet.interpolate.grid_sample(spectrograms, grid)

    else:
        spectrograms = None

    return features, target_pitch, periodicity, target_loudness, spectrograms


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

        # Automatic mixed precision
        with torch.autocast(device_type):
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
