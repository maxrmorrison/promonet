import contextlib
import os
from typing import List, Optional, Union
from pathlib import Path

import torch
import torchutil
import torchaudio
import tqdm

import promonet


###############################################################################
# Editing API
###############################################################################


def from_features(
    pitch: torch.Tensor,
    periodicity: torch.Tensor,
    loudness: torch.Tensor,
    ppg: torch.Tensor,
    speaker: Optional[Union[int, torch.Tensor]] = 0,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None) -> torch.Tensor:
    """Perform speech editing

    Args:
        pitch: The pitch contour
        periodicity: The periodicity contour
        loudness: The loudness contour
        ppg: The phonetic posteriorgram
        speaker: The speaker index
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        generated: The generated speech
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Maybe use a baseline method instead
    # TODO
    # if promonet.MODEL == 'psola':
    #     with torchutil.time.context('generate'):
    #         return promonet.baseline.psola.from_features(**locals())
    # elif promonet.MODEL == 'world':
    #     with torchutil.time.context('generate'):
    #         return promonet.baseline.world.from_features(**locals())

    # Generate
    return generate(
        pitch.to(device),
        periodicity.to(device),
        loudness.to(device),
        ppg.to(device),
        speaker,
        checkpoint)


def from_file(
    pitch_file: Union[str, os.PathLike],
    periodicity_file: Union[str, os.PathLike],
    loudness_file: Union[str, os.PathLike],
    ppg_file: Union[str, os.PathLike],
    speaker: Optional[Union[int, torch.Tensor]] = 0,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None) -> torch.Tensor:
    """Edit speech on disk

    Args:
        pitch_file: The pitch file
        periodicity_file: The periodicity file
        loudness_file: The loudness file
        ppg_file: The phonetic posteriorgram file
        speaker: The speaker index
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        generated: The generated speech
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Load features
    pitch = torch.load(pitch_file, map_location=device)
    periodicity = torch.load(periodicity_file, map_location=device)
    loudness = torch.load(loudness_file, map_location=device)
    ppg = torch.load(ppg_file, map_location=device)[None]

    # Maybe resample length
    ppg = promonet.interpolate.ppg(
        ppg,
        promonet.interpolate.grid.of_length(ppg, pitch.shape[-1]))

    # Generate
    try:
        return from_features(
            pitch,
            periodicity,
            loudness,
            ppg,
            speaker,
            checkpoint,
            gpu)
    except Exception:
        import pdb; pdb.set_trace()
        pass


def from_file_to_file(
    pitch_file: Union[str, os.PathLike],
    periodicity_file: Union[str, os.PathLike],
    loudness_file: Union[str, os.PathLike],
    ppg_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    speaker: Optional[Union[int, torch.Tensor]] = 0,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None):
    """Edit speech on disk and save to disk

    Args:
        pitch_file: The pitch file
        periodicity_file: The periodicity file
        loudness_file: The loudness file
        ppg_file: The phonetic posteriorgram file
        output_file: The file to save generated speech audio
        speaker: The speaker index
        checkpoint: The generator checkpoint
        gpu: The GPU index
    """
    # Generate
    generated = from_file(
        pitch_file,
        periodicity_file,
        loudness_file,
        ppg_file,
        speaker,
        checkpoint,
        gpu
    ).to(device='cpu', dtype=torch.float32)

    # Save
    output_file.parent.mkdir(exist_ok=True, parents=True)
    torchaudio.save(output_file, generated, promonet.SAMPLE_RATE)


def from_files_to_files(
    pitch_files: List[Union[str, os.PathLike]],
    periodicity_files: List[Union[str, os.PathLike]],
    loudness_files: List[Union[str, os.PathLike]],
    ppg_files: List[Union[str, os.PathLike]],
    output_files: List[Union[str, os.PathLike]],
    speakers: Optional[Union[List[int], torch.Tensor]] = None,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None):
    """Edit speech on disk and save to disk

    Args:
        pitch_files: The pitch files
        periodicity_files: The periodicity files
        loudness_files: The loudness files
        ppg_files: The phonetic posteriorgram files
        output_files: The files to save generated speech audio
        speakers: The speaker indices
        checkpoint: The generator checkpoint
        gpu: The GPU index
    """
    if speakers is None:
        speakers = [0] * len(pitch_files)

    # Generate
    iterator = zip(
        pitch_files,
        periodicity_files,
        loudness_files,
        ppg_files,
        output_files,
        speakers)
    for item in iterator:
        from_file_to_file(*item, checkpoint=checkpoint, gpu=gpu)


###############################################################################
# Generation pipeline
###############################################################################


def generate(
    pitch,
    periodicity,
    loudness,
    ppg,
    speaker=0,
    checkpoint=promonet.DEFAULT_CHECKPOINT):
    """Generate speech from phoneme and prosody features"""
    device = pitch.device

    with torchutil.time.context('load'):

        # Cache model
        if not hasattr(generate, 'model') or generate.device != device:
            model = promonet.model.Generator().to(device)
            if type(checkpoint) is str:
                checkpoint = Path(checkpoint)
            if checkpoint.is_dir():
                checkpoint = torchutil.checkpoint.latest_path(checkpoint)
            model, *_ = torchutil.checkpoint.load(checkpoint, model)
            generate.model = model
            generate.device = device

    with torchutil.time.context('generate'):

        # Default length is the entire sequence
        lengths = torch.tensor(
            (pitch.shape[-1],),
            dtype=torch.long,
            device=device)

        # Specify speaker
        speakers = torch.full(
            (1,),
            speaker,
            dtype=torch.long,
            device=device)

        # Generate
        with generation_context(generate.model):
            return generate.model(
                ppg,
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers)[0][0].cpu()


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
