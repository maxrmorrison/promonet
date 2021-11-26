import torch

import promovits


###############################################################################
# Phonetic posteriorgram
###############################################################################


def from_audio(audio, gpu=None):
    """Compute PPGs from audio"""
    # TODO
    pass


def from_file(audio_file, gpu=None):
    """Compute PPGs from audio file"""
    return from_audio(promovits.load.audio(audio_file), gpu)


def from_file_to_file(audio_file, output_file, gpu=None):
    """Compute PPGs from audio file and save to disk"""
    ppg = from_file(audio_file, gpu)
    torch.save(ppg, output_file)


def from_files_to_files(audio_files, output_files, gpu=None):
    """Compute PPGs from audio files and save to disk"""
    for audio_file, output_file in zip(audio_files, output_files):
        from_file_to_file(audio_file, output_file, gpu)
