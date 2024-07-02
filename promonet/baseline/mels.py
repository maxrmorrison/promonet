from pathlib import Path

import torch
import torchaudio
import torchutil

import promonet


###############################################################################
# Mel spectrogram reconstruction
###############################################################################


def from_audio(
    audio,
    sample_rate=promonet.SAMPLE_RATE,
    speaker=0,
    spectral_balance_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint=None,
    gpu=None
):
    """Perform Mel spectrogram reconstruction"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Resample
    audio = resample(audio.to(device), sample_rate)

    # Preprocess
    spectrogram = promonet.preprocess.spectrogram.from_audio(audio)

    # Reconstruct
    return from_features(
        spectrogram,
        speaker,
        spectral_balance_ratio,
        loudness_ratio,
        checkpoint)


def from_features(
    spectrogram,
    speaker=0,
    spectral_balance_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint=None
):
    """Perform Mel spectrogram reconstruction"""
    device = spectrogram.device

    with torchutil.time.context('load'):

        # Cache model
        if (
            not hasattr(from_features, 'model') or
            from_features.checkpoint != checkpoint or
            from_features.device != device
        ):
            model = promonet.model.MelGenerator().to(device)
            if type(checkpoint) is str:
                checkpoint = Path(checkpoint)
            if checkpoint.is_dir():
                checkpoint = torchutil.checkpoint.latest_path(
                    checkpoint,
                    'generator-*.pt')
            model, *_ = torchutil.checkpoint.load(checkpoint, model)
            from_features.model = model
            from_features.checkpoint = checkpoint
            from_features.device = device

    with torchutil.time.context('generate'):

        # Default length is the entire sequence
        lengths = torch.tensor(
            (spectrogram.shape[-1],),
            dtype=torch.long,
            device=device)

        # Specify speaker
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

        # Reconstruct
        with torchutil.inference.context(from_features.model):
            return from_features.model(
                spectrogram[None],
                speakers,
                spectral_balance_ratio,
                loudness_ratio
            )[0].to(torch.float32)


def from_file(
    audio_file,
    speaker=0,
    spectral_balance_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint=None,
    gpu=None
):
    """Perform Mel reconstruction from audio file"""
    return from_audio(
        promonet.load.audio(audio_file),
        speaker=speaker,
        spectral_balance_ratio=spectral_balance_ratio,
        loudness_ratio=loudness_ratio,
        checkpoint=checkpoint,
        gpu=gpu)


def from_file_to_file(
    audio_file,
    output_file,
    speaker=0,
    spectral_balance_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint=None,
    gpu=None
):
    """Perform Mel reconstruction from audio file and save"""
    # Reconstruct
    reconstructed = from_file(
        audio_file,
        speaker,
        spectral_balance_ratio,
        loudness_ratio,
        checkpoint,
        gpu)

    # Save
    torchaudio.save(output_file, reconstructed.cpu(), promonet.SAMPLE_RATE)


def from_files_to_files(
    audio_files,
    output_files,
    speakers=None,
    spectral_balance_ratio: float = 1.,
    loudness_ratio: float = 1.,
    checkpoint=None,
    gpu=None
):
    """Perform Mel reconstruction from audio files and save"""
    if speakers is None:
        speakers = [0] * len(audio_files)

    # Generate
    for item in zip(audio_files, output_files, speakers):
        from_file_to_file(
            *item,
            spectral_balance_ratio=spectral_balance_ratio,
            loudness_ratio=loudness_ratio,
            checkpoint=checkpoint,
            gpu=gpu)


###############################################################################
# Utilities
###############################################################################


def resample(audio, sample_rate):
    """Resample audio to ProMoNet sample rate"""
    # Cache resampling filter
    key = str(sample_rate)
    if not hasattr(resample, key):
        setattr(
            resample,
            key,
            torchaudio.transforms.Resample(sample_rate, promonet.SAMPLE_RATE))

    # Resample
    return getattr(resample, key)(audio)
