import matplotlib.pyplot as plt
import pyfoal
import pysodic

import promonet


###############################################################################
# Plot prosody
###############################################################################


def from_features(audio, pitch, periodicity, loudness, alignment):
    """Plot prosody from features"""
    figure, axes = plt.subplots(5, 1, figsize=(18, 6))

    # Plot audio
    axes[0].plot(audio.squeeze(), color='black', linewidth=.5)
    axes[0].set_axis_off()
    axes[0].set_ylim([-1., 1.])

    # Plot pitch
    axes[1].plot(pitch.squeeze(), color='black', linewidth=.5)
    axes[1].set_axis_off()

    # Plot periodicity
    axes[2].plot(periodicity.squeeze(), color='black', linewidth=.5)
    axes[2].set_axis_off()

    # Plot loudness
    axes[3].plot(loudness.squeeze(), color='black', linewidth=.5)
    axes[3].set_axis_off()

    # Plot alignment
    pyfoal.plot.phonemes(axes[4], alignment, 5)

    return figure


def from_file(text_file, audio_file, gpu=None):
    """Plot prosody from text and audio on disk"""
    # Load
    text = promonet.load.text(text_file)
    audio = promonet.load.audio(audio_file)

    # Plot
    return from_text_and_audio(text, audio, gpu)


def from_file_to_file(text_file, audio_file, output_file, gpu=None):
    """Plot prosody from text and audio on disk and save to disk"""
    # Plot
    figure = from_file(text_file, audio_file, gpu)

    # Save
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)


def from_text_and_audio(text, audio, gpu=None):
    """Plot prosody from text and audio inputs"""
    # Preprocess
    pitch, periodicity, loudness, _, alignment = pysodic.from_audio_and_text(
        audio,
        promonet.SAMPLE_RATE,
        text,
        promonet.HOPSIZE / promonet.SAMPLE_RATE,
        promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
        gpu=gpu)

    # Plot
    return from_features(audio, pitch, periodicity, loudness, alignment)
