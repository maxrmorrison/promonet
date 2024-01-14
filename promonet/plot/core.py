import matplotlib.pyplot as plt
import penn
import torch

import promonet


###############################################################################
# Plot speech representation
###############################################################################


def from_audio(
    audio,
    target_audio=None,
    features=promonet.DEFAULT_PLOT_FEATURES,
    gpu=None):
    """Plot speech representation from audio"""
    # Preprocess
    pitch, periodicity, loudness, ppg = promonet.preprocess.from_audio(
        audio,
        gpu=gpu)
    if target_audio is None:
        target_pitch = None
        target_periodicity = None
        target_loudness = None
        target_ppg = None
    else:
        (
            target_loudness,
            target_periodicity,
            target_pitch,
            target_ppg
        ) = promonet.preprocess.from_audio(audio, gpu=gpu)

    # Plot
    return from_features(
        audio,
        pitch,
        periodicity,
        loudness,
        ppg,
        target_pitch,
        target_periodicity,
        target_loudness,
        target_ppg,
        features=features)


def from_features(
        audio,
        pitch,
        periodicity,
        loudness,
        ppg,
        target_pitch=None,
        target_periodicity=None,
        target_loudness=None,
        target_ppg=None,
        features=promonet.DEFAULT_PLOT_FEATURES):
    """Plot speech representation"""
    figure, axes = plt.subplots(len(features), 1, figsize=(18, 2 * len(features)))

    # Plot audio
    i = 0
    if 'audio' in features:
        axes[i].plot(audio.squeeze().cpu(), color='black', linewidth=.5)
        axes[i].set_axis_off()
        axes[i].set_ylim([-1., 1.])
        i += 1

    # Plot pitch
    if 'pitch' in features:
        axes[i].plot(pitch.squeeze().cpu(), color='black', linewidth=1.)
        if target_pitch is not None:
            axes[i].plot(target_pitch.squeeze().cpu(), color='green', linewidth=1.)
            if target_periodicity is not None:
                voicing = penn.voicing.threshold(
                    periodicity,
                    promonet.VOICING_THRESHOLD)
                target_voicing = penn.voicing.threshold(
                    target_periodicity,
                    promonet.VOICING_THRESHOLD)
                cents = 1200 * torch.abs(torch.log2(pitch) - torch.log2(target_pitch))
                errors = (
                    voicing &
                    target_voicing &
                    (cents > promonet.ERROR_THRESHOLD_PITCH))
                pitch_errors = target_pitch.clone()
                pitch_errors[~errors] = float('nan')
                axes[i].plot(pitch_errors.squeeze().cpu(), color='red', linewidth=1.)
        axes[i].set_axis_off()
        i += 1

    # Plot periodicity
    if 'periodicity' in features:
        axes[i].plot(periodicity.squeeze().cpu(), color='black', linewidth=1.)
        if target_periodicity is not None:
            axes[i].plot(
                target_periodicity.squeeze().cpu(),
                color='green',
                linewidth=1.)
            errors = (
                torch.abs(periodicity - target_periodicity) >
                promonet.ERROR_THRESHOLD_PERIODICITY)
            periodicity_errors = target_periodicity.clone()
            periodicity_errors[~errors] = float('nan')
            axes[i].plot(
                periodicity_errors.squeeze().cpu(),
                color='red',
                linewidth=1.)
        axes[i].set_axis_off()
        i += 1

    # Plot loudness
    if 'loudness' in features:
        axes[i].plot(loudness.squeeze().cpu(), color='black', linewidth=1.)
        if target_loudness is not None:
            axes[i].plot(
                target_loudness.squeeze().cpu(),
                color='green',
                linewidth=1.)
            errors = (
                torch.abs(loudness - target_loudness) >
                promonet.ERROR_THRESHOLD_LOUDNESS)
            loudness_errors = target_loudness.clone()
            loudness_errors[~errors] = float('nan')
            axes[i].plot(
                loudness_errors.squeeze().cpu(),
                color='red',
                linewidth=1.)
        axes[i].set_axis_off()
        i += 1

    return figure


def from_file(
    audio_file,
    target_file=None,
    features=promonet.DEFAULT_PLOT_FEATURES,
    gpu=None):
    """Plot speech representation from audio on disk"""
    return from_audio(
        promonet.load.audio(audio_file),
        None if target_file is None else promonet.load.audio(target_file),
        features,
        gpu)


def from_file_to_file(
    audio_file,
    output_file,
    target_file=None,
    features=promonet.DEFAULT_PLOT_FEATURES,
    gpu=None):
    """Plot speech representation from audio on disk and save to disk"""
    # Plot
    figure = from_file(audio_file, target_file, features, gpu)

    # Save
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
