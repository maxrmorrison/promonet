import resampy
import soundfile
import torchutil

import promonet


###############################################################################
# Loudness data augmentation
###############################################################################


def from_audio(audio, sample_rate, ratio):
    """Perform volume data augmentation on audio"""
    # Augment audio
    augmented = promonet.preprocess.loudness.shift(
        audio,
        promonet.convert.ratio_to_db(ratio.item()))

    # Resample ratio if the audio clips
    while ((augmented <= -1.) | (augmented >= 1.)).any():
        ratio = promonet.data.augment.sample(1)[0]
        augmented = promonet.preprocess.loudness.shift(
            audio,
            promonet.convert.ratio_to_db(ratio.item()))

    # Resample to promonet sample rate
    augmented = resampy.resample(augmented, sample_rate, promonet.SAMPLE_RATE)

    return augmented, ratio


def from_file(audio_file, ratio):
    """Perform volume data augmentation on audio file"""
    return from_audio(*soundfile.read(str(audio_file)), ratio)


def from_file_to_file(audio_file, output_file, ratio):
    """Perform volume data augmentation on audio file and save"""
    augmented, new_ratio = from_file(audio_file, ratio)
    if new_ratio != ratio:
        output_file = (
            output_file.parent / output_file.name.replace(
                f'{int(ratio * 100):03d}',
                f'{int(new_ratio * 100):03d}'))
        ratio = new_ratio
    soundfile.write(str(output_file), augmented, promonet.SAMPLE_RATE)
    return ratio


def from_files_to_files(audio_files, output_files, ratios):
    """Perform volume data augmentation on audio files and save"""
    new_ratios = []
    for item in torchutil.iterator(
        zip(audio_files, output_files, ratios),
        'Augmenting loudness',
        total=len(audio_files)
    ):
        new_ratios.append(from_file_to_file(*item))
    return ratios


###############################################################################
# Loudness data augmentation
###############################################################################


def wrapper(item):
    return from_file_to_file(*item)
