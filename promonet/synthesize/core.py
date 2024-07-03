import os
from typing import List, Optional, Union
from pathlib import Path

import huggingface_hub
import torch
import torchaudio
import torchutil

import promonet


###############################################################################
# Editing API
###############################################################################


def from_features(
    loudness: torch.Tensor,
    pitch: torch.Tensor,
    periodicity: torch.Tensor,
    ppg: torch.Tensor,
    speaker: Union[int, torch.Tensor] = 0,
    spectral_balance_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> torch.Tensor:
    """Perform speech synthesis

    Args:
        loudness: The loudness contour
        pitch: The pitch contour
        periodicity: The periodicity contour
        ppg: The phonetic posteriorgram
        speaker: The speaker index or embedding
        spectral_balance_ratio: > 1 for Alvin and the Chipmunks; < 1 for Patrick Star
        loudness_ratio: > 1 for louder; < 1 for quieter
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        generated: The generated speech
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    if loudness.ndim == 2:
        loudness = loudness[None]

    return generate(
        loudness.to(device),
        pitch.to(device),
        periodicity.to(device),
        ppg.to(device),
        speaker,
        spectral_balance_ratio,
        loudness_ratio,
        checkpoint
    ).to(torch.float32)


def from_file(
    loudness_file: Union[str, os.PathLike],
    pitch_file: Union[str, os.PathLike],
    periodicity_file: Union[str, os.PathLike],
    ppg_file: Union[str, os.PathLike],
    speaker: Union[int, torch.Tensor, Path, str] = 0,
    spectral_balance_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> torch.Tensor:
    """Perform speech synthesis from features on disk

    Args:
        loudness_file: The loudness file
        pitch_file: The pitch file
        periodicity_file: The periodicity file
        ppg_file: The phonetic posteriorgram file
        speaker: The speaker index or embedding
        spectral_balance_ratio: > 1 for Alvin and the Chipmunks; < 1 for Patrick Star
        loudness_ratio: > 1 for louder; < 1 for quieter
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        generated: The generated speech
    """
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Load features
    loudness = torch.load(loudness_file)
    pitch = torch.load(pitch_file)
    periodicity = torch.load(periodicity_file)
    ppg = promonet.load.ppg(ppg_file, resample_length=pitch.shape[-1])[None]

    # Maybe load speaker embedding
    if promonet.ZERO_SHOT:
        speaker = torch.load(speaker).to(device)

    # Generate
    return from_features(
        loudness.to(device),
        pitch.to(device),
        periodicity.to(device),
        ppg.to(device),
        speaker,
        spectral_balance_ratio,
        loudness_ratio,
        checkpoint,
        gpu)


def from_file_to_file(
    loudness_file: Union[str, os.PathLike],
    pitch_file: Union[str, os.PathLike],
    periodicity_file: Union[str, os.PathLike],
    ppg_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    speaker: Union[int, torch.Tensor, Path, str] = 0,
    spectral_balance_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> None:
    """Perform speech synthesis from features on disk and save

    Args:
        loudness_file: The loudness file
        pitch_file: The pitch file
        periodicity_file: The periodicity file
        ppg_file: The phonetic posteriorgram file
        output_file: The file to save generated speech audio
        speaker: The speaker index or embedding
        spectral_balance_ratio: > 1 for Alvin and the Chipmunks; < 1 for Patrick Star
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
        spectral_balance_ratio,
        loudness_ratio,
        checkpoint,
        gpu
    ).to('cpu')

    # Save
    output_file.parent.mkdir(exist_ok=True, parents=True)
    torchaudio.save(output_file, generated, promonet.SAMPLE_RATE)


def from_files_to_files(
    loudness_files: List[Union[str, os.PathLike]],
    pitch_files: List[Union[str, os.PathLike]],
    periodicity_files: List[Union[str, os.PathLike]],
    ppg_files: List[Union[str, os.PathLike]],
    output_files: List[Union[str, os.PathLike]],
    speakers: Optional[Union[List[int], torch.Tensor, Path, str]] = None,
    spectral_balance_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> None:
    """Perform batched speech synthesis from features on disk and save

    Args:
        loudness_files: The loudness files
        pitch_files: The pitch files
        periodicity_files: The periodicity files
        ppg_files: The phonetic posteriorgram files
        output_files: The files to save generated speech audio
        speakers: The speaker indices or embeddings
        spectral_balance_ratio: > 1 for Alvin and the Chipmunks; < 1 for Patrick Star
        loudness_ratio: > 1 for louder; < 1 for quieter
        checkpoint: The generator checkpoint
        gpu: The GPU index
    """
    if speakers is None:
        speakers = [0] * len(pitch_files)

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
            spectral_balance_ratio=spectral_balance_ratio,
            loudness_ratio=loudness_ratio,
            checkpoint=checkpoint,
            gpu=gpu)


###############################################################################
# Pipeline
###############################################################################


def generate(
    loudness,
    pitch,
    periodicity,
    ppg,
    speaker=0,
    spectral_balance_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint=None
) -> torch.Tensor:
    """Generate speech from phoneme and prosody features"""
    device = pitch.device

    with torchutil.time.context('load'):

        # Cache model
        if (
            not hasattr(generate, 'model') or
            generate.checkpoint != checkpoint or
            generate.device != device
        ):
            if promonet.SPECTROGRAM_ONLY:
                model = promonet.model.MelGenerator().to(device)
            else:
                model = promonet.model.Generator().to(device)
            if checkpoint is None:
                checkpoint = huggingface_hub.hf_hub_download(
                    'maxrmorrison/promonet',
                    f'generator-00{promonet.STEPS}.pt')
            else:
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

        # Specify speaker
        if promonet.ZERO_SHOT:
            speakers = speaker.to(device)
        else:
            speakers = torch.full((1,), speaker, dtype=torch.long, device=device)

        # Format ratio
        spectral_balance_ratio = torch.tensor(
            [spectral_balance_ratio],
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
                speakers,
                spectral_balance_ratio,
                loudness_ratio,
                generate.model.default_previous_samples
            )[0]
