import os
from typing import List, Optional, Tuple, Union

import penn
import ppgs
import torch
import torchaudio

import promonet


###############################################################################
# Preprocess
###############################################################################


def from_audio(
    audio: torch.Tensor,
    sample_rate: int = promonet.SAMPLE_RATE,
    gpu: Optional[int] = None,
    features: list = ['loudness', 'pitch', 'periodicity', 'ppg'],
    loudness_bands: int = promonet.LOUDNESS_BANDS,
    max_harmonics=promonet.MAX_HARMONICS
) -> Tuple:
    """Preprocess audio

    Arguments
        audio: Audio to preprocess
        sample_rate: Audio sample rate
        gpu: The GPU index
        features: The features to preprocess.
            Options: [
                'loudness',
                'pitch',
                'periodicity',
                'ppg',
                'text',
                'harmonics',
                'speaker']
        loudness_bands: The number of A-weighted loudness bands
        max_harmonics: The maximum number of speech harmonics

    Returns
        loudness: The loudness contour
        pitch: The pitch contour
        periodicity: The periodicity contour
        ppg: The phonetic posteriorgram
        text: The text transcript
        harmonics: The speech harmonic contours
        speaker: The WavLM x-vector embeddings
    """
    result = []

    # Compute loudness
    if 'loudness' in features:
        device = f'cuda:{gpu}' if gpu is not None else 'cpu'
        result.append(
            promonet.preprocess.loudness.from_audio(
                audio,
                loudness_bands
            ).to(device))

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
        if 'pitch' in features:
            result.append(pitch)
        if 'periodicity' in features:
            result.append(periodicity)

    # Infer ppg
    if 'ppg' in features:
        ppg = ppgs.from_audio(audio, sample_rate, gpu=gpu)

        # Resample
        length = promonet.convert.samples_to_frames(
            torchaudio.functional.resample(
                audio.shape[-1],
                sample_rate,
                promonet.SAMPLE_RATE))
        ppg = promonet.edit.grid.sample(
            ppg,
            promonet.edit.grid.of_length(ppg, length),
            promonet.PPG_INTERP_METHOD)

        # Preserve distribution
        result.append(torch.softmax(torch.log(ppg + 1e-8), -2))

    # Infer transcript
    if 'text' in features:
        text = promonet.preprocess.text.from_audio(audio, sample_rate, gpu=gpu)
        result.append(text)

    # Compute harmonics
    if 'harmonics' in features:
        harmonics = promonet.preprocess.harmonics.from_audio(
            audio,
            sample_rate,
            max_harmonics=max_harmonics)
        result.append(harmonics)

    # Compute speaker embeddings
    if 'speaker' in features:
        speaker = promonet.preprocess.speaker.from_audio(
            audio,
            sample_rate,
            gpu=gpu)
        result.append(speaker)

    return (*result,)


def from_file(
    file: Union[str, bytes, os.PathLike],
    gpu: Optional[int] = None,
    features: list = ['loudness', 'pitch', 'periodicity', 'ppg'],
    loudness_bands: int = promonet.LOUDNESS_BANDS,
    max_harmonics=promonet.MAX_HARMONICS
) -> Tuple:
    """Preprocess audio on disk

    Arguments
        file: Audio file to preprocess
        gpu: The GPU index
        features: The features to preprocess.
            Options: [
                'loudness',
                'pitch',
                'periodicity',
                'ppg',
                'text',
                'harmonics',
                'speaker']
        loudness_bands: The number of A-weighted loudness bands
        max_harmonics: The maximum number of speech harmonics

    Returns
        loudness: The loudness contour
        pitch: The pitch contour
        periodicity: The periodicity contour
        ppg: The phonetic posteriorgram
        text: The text transcript
        harmonics: The speech harmonic contours
    """
    return from_audio(
        promonet.load.audio(file),
        gpu=gpu,
        features=features,
        loudness_bands=loudness_bands,
        max_harmonics=max_harmonics)


def from_file_to_file(
    file: Union[str, bytes, os.PathLike],
    output_prefix: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None,
    features: list = ['loudness', 'pitch', 'periodicity', 'ppg'],
    loudness_bands: int = promonet.LOUDNESS_BANDS,
    max_harmonics=promonet.MAX_HARMONICS
) -> None:
    """Preprocess audio on disk and save

    Arguments
        file: Audio file to preprocess
        output_prefix: File to save features, minus extension
        gpu: The GPU index
        features: The features to preprocess.
            Options: [
                'loudness',
                'pitch',
                'periodicity',
                'ppg',
                'text',
                'harmonics',
                'speaker']
        loudness_bands: The number of A-weighted loudness bands
        max_harmonics: The maximum number of speech harmonics
    """
    # Preprocess
    inferred_features = from_file(file, gpu, features, loudness_bands, max_harmonics)

    # Save
    if output_prefix is None:
        output_prefix = file.parent / file.stem
    if 'loudness' in features:
        torch.save(inferred_features[0], f'{output_prefix}-loudness.pt')
        del inferred_features[0]
    if 'pitch' in features:
        torch.save(inferred_features[0], f'{output_prefix}-pitch.pt')
        del inferred_features[0]
    if 'periodicity' in features:
        torch.save(inferred_features[0], f'{output_prefix}-periodicity.pt')
        del inferred_features[0]
    if 'ppg' in features:
        torch.save(
            inferred_features[0],
            f'{output_prefix}{ppgs.representation_file_extension()}')
        del inferred_features[0]
    if 'text' in features:
        with open(f'{output_prefix}.txt', 'w') as file:
            file.write(inferred_features[0])
        del inferred_features[0]
    if 'harmonics' in features:
        torch.save(inferred_features[0], f'{output_prefix}-harmonics.pt')
        del inferred_features[0]
    if 'speaker' in features:
        torch.save(inferred_features[0], f'{output_prefix}-speaker.pt')
        del inferred_features[0]


def from_files_to_files(
    files: List[Union[str, bytes, os.PathLike]],
    output_prefixes: Optional[List[Union[str, os.PathLike]]] = None,
    gpu: Optional[int] = None,
    features: list = ['loudness', 'pitch', 'periodicity', 'ppg'],
    loudness_bands: int = promonet.LOUDNESS_BANDS,
    max_harmonics=promonet.MAX_HARMONICS
) -> None:
    """Preprocess multiple audio files on disk and save

    Arguments
        files: Audio files to preprocess
        output_prefixes: Files to save features, minus extension
        gpu: The GPU index
        features: The features to preprocess.
            Options: [
                'loudness',
                'pitch',
                'periodicity',
                'ppg',
                'text',
                'harmonics',
                'speaker']
        loudness_bands: The number of A-weighted loudness bands
        max_harmonics: The maximum number of speech harmonics
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
            pitch_prefixes = [
                f'{prefix}-viterbi' for prefix in output_prefixes]
        else:
            decoder = 'argmax'
            voicing_threshold = promonet.VOICING_THRESHOLD
            pitch_prefixes = output_prefixes
        penn.from_files_to_files(
            files,
            pitch_prefixes,
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
        promonet.preprocess.loudness.from_files_to_files(
            files,
            [f'{prefix}-loudness.pt' for prefix in output_prefixes],
            bands=loudness_bands)

    # Infer transcript
    if 'text' in features:
        promonet.preprocess.text.from_files_to_files(
            files,
            [f'{prefix}.txt' for prefix in output_prefixes],
            gpu)

    # Compute harmonics
    if 'harmonics' in features:
        promonet.preprocess.harmonics.from_files_to_files(
            files,
            [f'{prefix}-harmonics.pt' for prefix in output_prefixes],
            pitch_files=[f'{prefix}-pitch.pt' for prefix in pitch_prefixes],
            output_feature_files=[
                f'{prefix}-harmonicfeatures.pt' for prefix in output_prefixes],
            gpu=gpu,
            max_harmonics=max_harmonics)

    # Compute speaker embeddings
    if 'speaker' in features:
        promonet.preprocess.speaker.from_files_to_files(
            files,
            [f'{prefix}-speaker.pt' for prefix in output_prefixes],
            gpu=gpu)
