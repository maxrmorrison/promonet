import matplotlib
import matplotlib.pyplot as plt
import penn
import ppgs
import torch

import promonet


###############################################################################
# Plot speech representation
###############################################################################


def from_audio(
    audio,
    target_audio=None,
    features=promonet.DEFAULT_PLOT_FEATURES,
    gpu=None
):
    """Plot speech representation from audio"""
    # Preprocess
    loudness, pitch, periodicity, ppg = promonet.preprocess.from_audio(
        audio,
        features=features,
        gpu=gpu)
    loudness = promonet.loudness.band_average(loudness, 1)
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
        target_loudness = promonet.loudness.band_average(target_loudness, 1)

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
    features=promonet.DEFAULT_PLOT_FEATURES
):
    """Plot speech representation"""
    height_ratios = [3 * (feature == 'ppg') + 1 for feature in features]
    figure, axes = plt.subplots(
        len(features),
        1,
        figsize=(8, 2 * len(features)),
        gridspec_kw={'height_ratios': height_ratios})

    # Plot audio
    i = 0
    for feature in features:
        if feature == 'audio':
            axes[i].plot(audio.squeeze().cpu(), color='black', linewidth=.5)
            axes[i].set_xmargin(0.)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].set_xticks([])
            axes[i].set_ylim([-1., 1.])
            axes[i].tick_params(axis=u'both', which=u'both', length=0)
            # axes[i].set_ylabel('Audio', fontsize=12)
            i += 1

        # Plot PPGs
        if feature == 'ppg':
            ppg = ppg.squeeze()
            probable = ppg > .05
            if target_ppg is not None:
                target_ppg = target_ppg.squeeze()
                probable = probable | (target_ppg > .05)
            used = probable.sum(-1) > 0
            ppg = ppg[used]
            ppg[ppg < .05] = 0.

            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'cmap',
                ['none', 'blue'],
                256)
            cmap._init()
            alphas = torch.linspace(0, 1. if target_ppg is None else .5, cmap.N + 3)
            cmap._lut[:, -1] = alphas

            if target_ppg is not None:
                target_ppg = target_ppg[used]
                target_ppg[target_ppg < .05] = 0.
                target_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'target_cmap',
                    ['none', 'red'],
                    256)
                target_cmap._init()
                target_cmap._lut[:, -1] = alphas
                axes[i].imshow(target_ppg.cpu(), aspect='auto', interpolation='none', cmap=target_cmap)

            axes[i].imshow(ppg.cpu(), aspect='auto', interpolation='none', cmap=cmap)
            axes[i].set_xmargin(0.)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].set_ylabel('Phonetic posteriorgram', labelpad=-18, fontsize=12)
            axes[i].set_xticks([])
            axes[i].tick_params(axis=u'both', which=u'both', length=0)
            yticks = torch.arange(len(ppg))
            axes[i].set_yticks(yticks, [ppgs.PHONEMES[i] for i, u in enumerate(used) if u])
            for tick in (yticks - .5).tolist() + [yticks[-1] + .5]:
                axes[i].hlines(tick, xmin=0., xmax=ppg.shape[-1], color='#aaaa', linestyle='--')
            i += 1

        # Plot pitch
        if feature == 'pitch':
            axes[i].plot(pitch.squeeze().cpu(), color='black', linewidth=1.)
            ymin, ymax = pitch.min(), pitch.max()
            if target_pitch is not None:
                axes[i].plot(target_pitch.squeeze().cpu(), color='green', linewidth=1.)
                ymin = min(target_pitch.min(), ymin)
                ymax = max(target_pitch.max(), ymax)
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
            axes[i].set_xmargin(0.)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].set_xticks([])
            axes[i].tick_params(axis=u'both', which=u'both', length=0)
            ymin = ymin // 50 * 50
            ymax = (ymax + 50) // 50 * 50
            diff = ymax - ymin
            if diff <= 100:
                ystep = 25
            elif diff <= 150:
                ystep = 50
            else:
                ystep = 100
            yticks = torch.arange(ymin, ymax + ystep, ystep)
            axes[i].set_yticks(yticks)
            for tick in yticks:
                axes[i].hlines(tick, xmin=0., xmax=pitch.shape[-1], color='#aaaa', linestyle='--')
            axes[i].set_ylabel('Pitch', fontsize=12, labelpad=14 if pitch.max() < 100. else None)
            i += 1

        # Plot periodicity
        if feature == 'periodicity':
            axes[i].plot(periodicity.squeeze().cpu(), color='black', linewidth=1.)
            ymin, ymax = periodicity.min(), periodicity.max()
            if target_periodicity is not None:
                ymin = min(target_periodicity.min(), ymin)
                ymax = max(target_periodicity.max(), ymax)
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
            axes[i].set_xmargin(0.)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].set_xticks([])
            axes[i].tick_params(axis=u'both', which=u'both', length=0)
            ymin = ymin // .25 * .25
            ymax = (ymax + .25) // .25 * .25
            ystep = .25
            yticks = torch.arange(ymin, ymax + ystep, ystep)
            axes[i].set_yticks(yticks)
            for tick in yticks:
                axes[i].hlines(tick, xmin=0., xmax=periodicity.shape[-1], color='#aaaa', linestyle='--')
            axes[i].set_ylabel('Periodicity', fontsize=12)
            i += 1

        # Plot loudness
        if feature == 'loudness':
            axes[i].plot(loudness.squeeze().cpu(), color='black', linewidth=1.)
            ymin, ymax = loudness.min(), loudness.max()
            if target_loudness is not None:
                ymin = min(target_loudness.min(), ymin)
                ymax = max(target_loudness.max(), ymax)
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
            axes[i].set_xmargin(0.)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].set_xticks([])
            axes[i].tick_params(axis=u'both', which=u'both', length=0)
            ymin = ymin // 5 * 5
            ymax = (ymax + 5) // 5 * 5
            ystep = 10 if (ymax - ymin <= 30) else 20
            yticks = torch.arange(ymin, ymax + ystep, ystep)
            axes[i].set_yticks(yticks)
            for tick in yticks:
                axes[i].hlines(tick, xmin=0., xmax=loudness.shape[-1], color='#aaaa', linestyle='--')
            axes[i].set_ylabel('Loudness', fontsize=12)
            i += 1

    return figure


def from_file(
    audio_file,
    target_file=None,
    features=promonet.DEFAULT_PLOT_FEATURES,
    gpu=None
):
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
    gpu=None
):
    """Plot speech representation from audio on disk and save to disk"""
    # Plot
    figure = from_file(audio_file, target_file, features, gpu)

    # Save
    figure.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
