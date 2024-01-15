import os
from typing import List, Optional, Tuple, Union

import penn
import ppgs
import torch
import whisper

import promonet


###############################################################################
# Preprocess
###############################################################################


def from_audio(
    audio: torch.Tensor,
    sample_rate: int = promonet.SAMPLE_RATE,
    gpu: Optional[int] = None,
    features: list = ['loudness', 'periodicity', 'pitch', 'ppg']
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]
]:
    """Preprocess audio

    Arguments
        audio: Audio to preprocess
        sample_rate: Audio sample rate
        gpu: The GPU index
        features: The features to preprocess.
            Options: ['loudness', 'periodicity', 'pitch', 'ppg', 'text'].

    Returns
        loudness: The loudness contour
        periodicity: The periodicity contour
        pitch: The pitch contour
        ppg: The phonetic posteriorgram
        text: The text transcript
    """
    result = []

    # Compute loudness
    if 'loudness' in features:
        device = f'cuda:{gpu}' if gpu is not None else 'cpu'
        result.append(promonet.loudness.from_audio(audio).to(device))

    # Estimate pitch and periodicity
    if 'pitch' in features or 'periodicity' in features:
        if promonet.VITERBI_DECODE_PITCH:
            decoder = 'viterbi'
            voicing_threshold = None
        else:
            decoder = 'argmax'
            voicing_threshold = promonet.VOICING_THRESHOLD
        pitch, periodicity = penn.from_audio(
            audio,
            sample_rate=sample_rate,
            hopsize=promonet.convert.samples_to_seconds(promonet.HOPSIZE),
            fmin=promonet.FMIN,
            fmax=promonet.FMAX,
            batch_size=2048,
            center='half-hop',
            decoder=decoder,
            interp_unvoiced_at=voicing_threshold,
            gpu=gpu)
        result.extend([periodicity, pitch])

    # Infer ppg
    if 'ppg' in features:
        ppg = ppgs.from_audio(audio, sample_rate, gpu=gpu)
        ppg = promonet.load.ppg(ppg, resample_length=pitch.shape[-1])
        result.append(ppg)

    # Infer transcript
    if 'text' in features:
        text = promonet.preprocess.text.from_audio(audio, sample_rate, gpu=gpu)
        result.append(text)

    return (*result,)


def from_file(
    file: Union[str, bytes, os.PathLike],
    gpu: Optional[int] = None,
    features: list = ['loudness', 'periodicity', 'pitch', 'ppg']
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]
]:
    """Preprocess audio on disk

    Arguments
        file: Audio file to preprocess
        gpu: The GPU index
        features: The features to preprocess.
            Options: ['loudness', 'periodicity', 'pitch', 'ppg', 'text'].

    Returns
        loudness: The loudness contour
        periodicity: The periodicity contour
        pitch: The pitch contour
        ppg: The phonetic posteriorgram
        text: The text transcript
    """
    return from_audio(promonet.load.audio(file), gpu=gpu, features=features)


def from_file_to_file(
    file: Union[str, bytes, os.PathLike],
    output_prefix: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None,
    features: list = ['loudness', 'periodicity', 'pitch', 'ppg']
) -> None:
    """Preprocess audio on disk and save

    Arguments
        file: Audio file to preprocess
        output_prefix: File to save features, minus extension
        gpu: The GPU index
        features: The features to preprocess.
            Options: ['loudness', 'periodicity', 'pitch', 'ppg', 'text'].
    """
    # Preprocess
    features = from_file(file, gpu, features)

    # Save
    if output_prefix is None:
        output_prefix = file.parent / file.stem
    if 'loudness' in features:
        torch.save(features[0], f'{output_prefix}-loudness.pt')
        del features[0]
    if 'periodicity' in features:
        torch.save(features[0], f'{output_prefix}-periodicity.pt')
        del features[0]
    if 'pitch' in features:
        torch.save(features[0], f'{output_prefix}-pitch.pt')
        del features[0]
    if 'ppg' in features:
        torch.save(
            features[0],
            f'{output_prefix}{ppgs.representation_file_extension()}')
        del features[0]
    if 'text' in features:
        with open(f'{output_prefix}.txt', 'w') as file:
            file.write(features[0])


def from_files_to_files(
    files: List[Union[str, bytes, os.PathLike]],
    output_prefixes: Optional[List[Union[str, os.PathLike]]] = None,
    gpu: Optional[int] = None,
    features: list = ['loudness', 'periodicity', 'pitch', 'ppg']
) -> None:
    """Preprocess multiple audio files on disk and save

    Arguments
        files: Audio files to preprocess
        output_prefixes: Files to save features, minus extension
        gpu: The GPU index
        features: The features to preprocess.
            Options: ['loudness', 'periodicity', 'pitch', 'ppg', 'text'].
    """
    if output_prefixes is None:
        output_prefixes = [file.parent / file.stem for file in files]

    # Preprocess phonetic posteriorgrams
    extension = ppgs.representation_file_extension()
    if 'ppg' in features:
        ppgs.from_files_to_files(
            files,
            [f'{prefix}{extension}' for prefix in output_prefixes],
            num_workers=promonet.NUM_WORKERS,
            max_frames=5000,
            gpu=gpu)

    # Preprocess pitch and periodicity
    if 'pitch' in features or 'periodicity' in features:
        if promonet.VITERBI_DECODE_PITCH:
            decoder = 'viterbi'
            voicing_threshold = None
            output_prefixes = [
                f'{prefix}-viterbi' for prefix in output_prefixes]
        else:
            decoder = 'argmax'
            voicing_threshold = promonet.VOICING_THRESHOLD
        penn.from_files_to_files(
            files,
            output_prefixes,
            hopsize=promonet.convert.samples_to_seconds(promonet.HOPSIZE),
            fmin=promonet.FMIN,
            fmax=promonet.FMAX,
            batch_size=2048,
            center='half-hop',
            decoder=decoder,
            interp_unvoiced_at=voicing_threshold,
            gpu=gpu)

    # Preprocess loudness
    if 'loudness' in features:
        promonet.loudness.from_files_to_files(
            files,
            [f'{prefix}-loudness.pt' for prefix in output_prefixes])

    # Infer transcript
    if 'text' in features:
        promonet.preprocess.text.from_files_to_files(
            files,
            [f'{prefix}.txt' for prefix in output_prefixes],
            gpu)
