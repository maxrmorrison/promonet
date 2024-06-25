import os
from typing import List, Optional, Union
from pathlib import Path

import torch
import torchaudio
import torchutil

import promonet


###############################################################################
# Editing API
###############################################################################


def from_features(
    loudness: Optional[torch.Tensor] = None,
    pitch: Optional[torch.Tensor] = None,
    periodicity: Optional[torch.Tensor] = None,
    ppg: Optional[torch.Tensor] = None,
    speaker: Optional[Union[int, torch.Tensor]] = 0,
    formant_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None
) -> torch.Tensor:
    """Perform speech synthesis

    Args:
        loudness: The loudness contour
        pitch: The pitch contour
        periodicity: The periodicity contour
        ppg: The phonetic posteriorgram
        speaker: The speaker index
        formant_ratio: > 1 for Alvin and the Chipmunks; < 1 for Patrick Star
        loudness_ratio: > 1 for louder; < 1 for quieter
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        generated: The generated speech
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    return generate(
        loudness.to(device) if loudness is not None else None,
        pitch.to(device) if pitch is not None else None,
        periodicity.to(device) if periodicity is not None else None,
        ppg.to(device) if ppg is not None else None,
        speaker,
        formant_ratio,
        loudness_ratio,
        checkpoint
    ).to(torch.float32)


def from_file(
    loudness_file: Optional[Union[str, os.PathLike]] = None,
    pitch_file: Optional[Union[str, os.PathLike]] = None,
    periodicity_file: Optional[Union[str, os.PathLike]] = None,
    ppg_file: Optional[Union[str, os.PathLike]] = None,
    speaker: Optional[Union[int, torch.Tensor]] = 0,
    formant_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None
) -> torch.Tensor:
    """Perform speech synthesis from features on disk

    Args:
        loudness_file: The loudness file
        pitch_file: The pitch file
        periodicity_file: The periodicity file
        ppg_file: The phonetic posteriorgram file
        speaker: The speaker index
        formant_ratio: > 1 for Alvin and the Chipmunks; < 1 for Patrick Star
        loudness_ratio: > 1 for louder; < 1 for quieter
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        generated: The generated speech
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Load features
    loudness = torch.load(loudness_file).to(device) if loudness_file is not None else None
    pitch = torch.load(pitch_file).to(device) if pitch_file is not None else None
    periodicity = torch.load(periodicity_file).to(device) if periodicity_file is not None else None
    ppg = promonet.load.ppg(ppg_file, resample_length=pitch.shape[-1])[None].to(device) if ppg_file is not None else None

    # Generate
    return from_features(
        loudness,
        pitch,
        periodicity,
        ppg,
        speaker,
        formant_ratio,
        loudness_ratio,
        checkpoint,
        gpu)


def from_file_to_file(
    loudness_file: Optional[Union[str, os.PathLike]] = None,
    pitch_file: Optional[Union[str, os.PathLike]] = None,
    periodicity_file: Optional[Union[str, os.PathLike]] = None,
    ppg_file: Optional[Union[str, os.PathLike]] = None,
    output_file: Optional[Union[str, os.PathLike]] = None,
    speaker: Optional[Union[int, torch.Tensor]] = 0,
    formant_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None
) -> None:
    """Perform speech synthesis from features on disk and save

    Args:
        loudness_file: The loudness file
        pitch_file: The pitch file
        periodicity_file: The periodicity file
        ppg_file: The phonetic posteriorgram file
        output_file: The file to save generated speech audio
        speaker: The speaker index
        formant_ratio: > 1 for Alvin and the Chipmunks; < 1 for Patrick Star
        loudness_ratio: > 1 for louder; < 1 for quieter
        checkpoint: The generator checkpoint
        gpu: The GPU index
    """
    # Generate
    generated = from_file(
        loudness_file,
        pitch_file,
        periodicity_file,
        ppg_file,
        speaker,
        formant_ratio,
        loudness_ratio,
        checkpoint,
        gpu
    ).to('cpu')

    # Save
    if output_file is None:
        raise ValueError('output_file cannot be None')
    output_file.parent.mkdir(exist_ok=True, parents=True)
    torchaudio.save(output_file, generated, promonet.SAMPLE_RATE)


def from_files_to_files(
    loudness_files: List[Union[str, os.PathLike]] = None,
    pitch_files: List[Union[str, os.PathLike]] = None,
    periodicity_files: List[Union[str, os.PathLike]] = None,
    ppg_files: List[Union[str, os.PathLike]] = None,
    output_files: List[Union[str, os.PathLike]] = None,
    speakers: Optional[Union[List[int], torch.Tensor]] = None,
    formant_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None
) -> None:
    """Perform batched speech synthesis from features on disk and save

    Args:
        loudness_files: The loudness files
        pitch_files: The pitch files
        periodicity_files: The periodicity files
        ppg_files: The phonetic posteriorgram files
        output_files: The files to save generated speech audio
        speakers: The speaker indices
        formant_ratio: > 1 for Alvin and the Chipmunks; < 1 for Patrick Star
        loudness_ratio: > 1 for louder; < 1 for quieter
        checkpoint: The generator checkpoint
        gpu: The GPU index
    """
    if speakers is None:
        speakers = [0] * len(pitch_files)

    if output_files is None:
        raise ValueError('output_files cannot be None')

    num_files = len(output_files)

    if loudness_files is None:
        loudness_files = [None] * num_files
    if pitch_files is None:
        pitch_files = [None] * num_files
    if periodicity_files is None:
        periodicity_files = [None] * num_files
    if ppg_files is None:
        ppg_files = [None] * num_files

    # Generate
    iterator = zip(
        loudness_files,
        pitch_files,
        periodicity_files,
        ppg_files,
        output_files,
        speakers)
    for item in iterator:
        from_file_to_file(
            *item,
            formant_ratio=formant_ratio,
            loudness_ratio=loudness_ratio,
            checkpoint=checkpoint,
            gpu=gpu)


###############################################################################
# Pipeline
###############################################################################


def generate(
    loudness=None,
    pitch=None,
    periodicity=None,
    ppg=None,
    speaker=0,
    formant_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint=promonet.DEFAULT_CHECKPOINT
) -> torch.Tensor:
    """Generate speech from phoneme and prosody features"""
    if pitch is None:
        raise ValueError('pitch cannot be None')
    device = pitch.device

    with torchutil.time.context('load'):

        # Cache model
        if (
            not hasattr(generate, 'model') or
            generate.checkpoint != checkpoint or
            generate.device != device
        ):
            model = promonet.model.Generator().to(device)
            if type(checkpoint) is str:
                checkpoint = Path(checkpoint)
            if checkpoint.is_dir():
                checkpoint = torchutil.checkpoint.latest_path(
                    checkpoint,
                    'generator-*.pt')
            model, *_ = torchutil.checkpoint.load(checkpoint, model)
            generate.model = model
            generate.checkpoint = checkpoint
            generate.device = device

    with torchutil.time.context('generate'):

        # Default length is the entire sequence
        lengths = torch.tensor(
            (pitch.shape[-1],),
            dtype=torch.long,
            device=device)

        # Specify speaker
        speakers = torch.full((1,), speaker, dtype=torch.long, device=device)

        # Format ratio
        formant_ratio = torch.tensor(
            [formant_ratio],
            dtype=torch.float,
            device=device)

        # Loudness ratio
        loudness_ratio = torch.tensor(
            [loudness_ratio],
            dtype=torch.float,
            device=device)

        # Generate
        with torchutil.inference.context(generate.model):
            return generate.model(
                loudness,
                pitch,
                periodicity,
                ppg,
                lengths,
                speakers,
                formant_ratio,
                loudness_ratio
            )[0][0]
