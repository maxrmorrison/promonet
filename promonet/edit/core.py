import torch

import promonet


def from_features(
    pitch,
    periodicity,
    loudness,
    ppgs,
    pitch_shift_cents=None,
    time_stretch_ratio=None,
    loudness_scale_db=None
):
    """Edit speech representation"""
    # Maybe time-stretch
    if time_stretch_ratio is not None:

        # Create time-stretch grid
        # TODO - make grid based on V/UV in PPGs
        grid = promonet.interpolate.grid.constant(
            ppgs,
            time_stretch_ratio)

        # Time-stretch
        pitch = promonet.interpolate.pitch(pitch, grid)
        periodicity = promonet.interpolate.grid_sample(periodicity, grid)
        loudness = promonet.interpolate.grid_sample(loudness, grid)
        # TODO - SLERP
        ppgs = promonet.interpolate.ppgs(ppgs, grid)

    # Maybe pitch-shift
    if pitch_shift_cents is not None:
        pitch *= promonet.convert.cents_to_ratio(pitch_shift_cents)
        pitch = torch.clip(pitch, promonet.FMIN, promonet.FMAX)

    # Maybe loudness-scale
    if loudness_scale_db is not None:
        loudness += loudness_scale_db

    return pitch, periodicity, loudness, ppgs


def from_file(
    pitch_file,
    periodicity_file,
    loudness_file,
    ppgs_file,
    pitch_shift_cents=None,
    time_stretch_ratio=None,
    loudness_scale_db=None
):
    """Edit speech representation on disk"""
    return from_features(
        torch.load(pitch_file),
        torch.load(periodicity_file),
        torch.load(loudness_file),
        torch.load(ppgs_file),
        pitch_shift_cents,
        time_stretch_ratio,
        loudness_scale_db)


def from_file_to_file(
    pitch_file,
    periodicity_file,
    loudness_file,
    ppgs_file,
    output_prefix,
    pitch_shift_cents=None,
    time_stretch_ratio=None,
    loudness_scale_db=None
):
    """Edit speech representation on disk and save to disk"""
    # Edit
    pitch, periodicity, loudness, ppgs = from_file(
        pitch_file,
        periodicity_file,
        loudness_file,
        ppgs_file,
        pitch_shift_cents,
        time_stretch_ratio,
        loudness_scale_db)

    # Save
    torch.save(pitch, f'{output_prefix}-pitch.pt')
    torch.save(periodicity, f'{output_prefix}-periodicity.pt')
    torch.save(loudness, f'{output_prefix}-loudness.pt')
    torch.save(ppgs, f'{output_prefix}-ppg.pt')


def from_files_to_files(
    pitch_files,
    periodicity_files,
    loudness_files,
    ppgs_files,
    output_prefixes,
    pitch_shift_cents=None,
    time_stretch_ratio=None,
    loudness_scale_db=None
):
    """Edit speech representations on disk and save to disk"""
    for pitch_file, periodicity_file, loudness_file, ppgs_file, prefix in zip(
        pitch_files,
        periodicity_files,
        loudness_files,
        ppgs_files,
        output_prefixes
    ):
        from_file_to_file(
            pitch_file,
            periodicity_file,
            loudness_file,
            ppgs_file,
            prefix,
            pitch_shift_cents,
            time_stretch_ratio,
            loudness_scale_db)
