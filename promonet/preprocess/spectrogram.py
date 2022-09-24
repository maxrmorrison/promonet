import functools
import multiprocessing as mp

import torch
import librosa

import promonet


###############################################################################
# Spectrogram computation
###############################################################################


def from_audio(audio, mels=False):
    """Compute spectrogram from audio"""
    # Cache hann window
    if (
        not hasattr(from_audio, 'window') or
        from_audio.dtype != audio.dtype or
        from_audio.device != audio.device
    ):
        from_audio.window = torch.hann_window(
            promonet.WINDOW_SIZE,
            dtype=audio.dtype,
            device=audio.device)
        from_audio.dtype = audio.dtype
        from_audio.device = audio.device

    # Pad audio
    size = (promonet.NUM_FFT - promonet.HOPSIZE) // 2
    audio = torch.nn.functional.pad(
        audio,
        (size, size),
        mode='reflect')

    # Compute stft
    stft = torch.stft(
        audio.squeeze(1),
        promonet.NUM_FFT,
        hop_length=promonet.HOPSIZE,
        window=from_audio.window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True)
    stft = torch.view_as_real(stft)

    # Compute magnitude
    spectrogram = torch.sqrt(stft.pow(2).sum(-1) + 1e-6)

    # Maybe convert to mels
    spectrogram = linear_to_mel(spectrogram) if mels else spectrogram

    return spectrogram.squeeze(0)


def from_file(audio_file, mels=False):
    """Compute spectrogram from audio file"""
    audio = promonet.load.audio(audio_file)
    return from_audio(audio, mels)


def from_file_to_file(audio_file, output_file, mels=False):
    """Compute spectrogram from audio file and save to disk"""
    output = from_file(audio_file, mels)
    torch.save(output, output_file)


def from_files_to_files(audio_files, output_files, mels=False):
    """Compute spectrogram from audio files and save to disk"""
    # TODO - this multiprocessing fails
    # with mp.get_context('spawn').Pool() as pool:
    #     pool.starmap(preprocess_fn, zip(audio_files, output_files))
    preprocess_fn = functools.partial(from_file_to_file, mels=mels)
    for item in zip(audio_files, output_files):
        preprocess_fn(*item)


###############################################################################
# Utilities
###############################################################################


def linear_to_mel(spectrogram):
    # Create mel basis
    if not hasattr(linear_to_mel, 'mel_basis'):
        basis = librosa.filters.mel(
            sr=promonet.SAMPLE_RATE,
            n_fft=promonet.NUM_FFT,
            n_mels=promonet.NUM_MELS)
        basis = torch.from_numpy(basis)
        basis = basis.to(spectrogram.dtype).to(spectrogram.device)
        linear_to_mel.basis = basis

    # Convert to mels
    melspectrogram = torch.matmul(linear_to_mel.basis, spectrogram)

    # Apply dynamic range compression
    return torch.log(torch.clamp(melspectrogram, min=1e-5))
