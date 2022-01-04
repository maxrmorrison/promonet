import matplotlib.pyplot as plt

import promovits


###############################################################################
# Plotting utilities
###############################################################################


def alignment(x, info=None):
    """Plot alignment"""
    figure, axis = plt.subplots(figsize=(6, 4))
    image = axis.imshow(
        x.transpose(),
        aspect='auto',
        origin='lower',
        interpolation='none')
    figure.colorbar(image, ax=axis)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    return figure


def spectrogram(x):
    """Plot spectrogram"""
    figure, axis = plt.subplots(figsize=(10, 2))
    image = axis.imshow(x, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(image, ax=axis)
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()
    return figure


def spectrogram_from_audio(audio):
    """Plot spectrogram given audio signal"""
    mels = promovits.preprocess.spectrogram.from_audio(audio.float(), True)
    return spectrogram(mels.cpu().numpy())
