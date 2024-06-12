import resampy
import soundfile
import torchutil

import promonet


###############################################################################
# Pitch-shifting data augmentation
###############################################################################


def from_audio(audio, sample_rate, ratio):
    """Perform pitch-shifting data augmentation on audio"""
    # Augment audio
    augmented = resampy.resample(audio, int(ratio * sample_rate), sample_rate)

    # Resample to promonet sample rate
    return resampy.resample(augmented, sample_rate, promonet.SAMPLE_RATE)


def from_file(audio_file, ratio):
    """Perform pitch-shifting data augmentation on audio file"""
    return from_audio(*soundfile.read(str(audio_file)), ratio)


def from_file_to_file(audio_file, output_file, ratio):
    """Perform pitch-shifting data augmentation on audio file and save"""
    augmented = from_file(audio_file, ratio)
    soundfile.write(str(output_file), augmented, promonet.SAMPLE_RATE)


def from_files_to_files(audio_files, output_files, ratios):
    """Perform pitch-shifting data augmentation on audio files and save"""
    torchutil.multiprocess_iterator(
        wrapper,
        zip(audio_files, output_files, ratios),
        'Augmenting pitch',
        total=len(audio_files),
        num_workers=promonet.NUM_WORKERS)


###############################################################################
# Utilities
###############################################################################


def wrapper(item):
    from_file_to_file(*item)
