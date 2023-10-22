import os
from typing import List, Optional, Tuple, Union

import torch

import promonet


###############################################################################
# Edit speech features
###############################################################################


def from_features(
    pitch: torch.Tensor,
    periodicity: torch.Tensor,
    loudness: torch.Tensor,
    ppg: torch.Tensor,
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edit speech representation

    Arguments
        pitch: Pitch contour to edit
        periodicity: Periodicity contour to edit
        loudness: Loudness contour to edit
        ppg: PPG to edit
        pitch_shift_cents: Amount of pitch-shifting in cents
        time_stretch_ratio: Amount of time-stretching. Faster when above one.
        loudness_scale_db: Amount of loudness scaling in dB

    Returns
        edited_pitch, edited_periodicity, edited_loudness, edited_ppg
    """
    # Maybe time-stretch
    if time_stretch_ratio is not None:

        # Create time-stretch grid
        # TODO - voiced-only interpolation from PPGs
        grid = promonet.interpolate.grid.constant(
            ppg,
            time_stretch_ratio)

        # Time-stretch
        pitch = promonet.interpolate.pitch(pitch, grid)
        periodicity = promonet.interpolate.grid_sample(periodicity, grid)
        loudness = promonet.interpolate.grid_sample(loudness, grid)
        ppg = promonet.interpolate.ppg(ppg, grid)

    # Maybe pitch-shift
    if pitch_shift_cents is not None:
        pitch *= promonet.convert.cents_to_ratio(pitch_shift_cents)
        pitch = torch.clip(pitch, promonet.FMIN, promonet.FMAX)

    # Maybe loudness-scale
    if loudness_scale_db is not None:
        loudness += loudness_scale_db

    return pitch, periodicity, loudness, ppg


def from_file(
    pitch_file: Union[str, bytes, os.PathLike],
    periodicity_file: Union[str, bytes, os.PathLike],
    loudness_file: Union[str, bytes, os.PathLike],
    ppg_file: Union[str, bytes, os.PathLike],
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edit speech representation on disk

    Arguments
        pitch_file: Pitch file to edit
        periodicity_file: Periodicity file to edit
        loudness_file: Loudness file to edit
        ppg_file: PPG file to edit
        pitch_shift_cents: Amount of pitch-shifting in cents
        time_stretch_ratio: Amount of time-stretching. Faster when above one.
        loudness_scale_db: Amount of loudness scaling in dB

    Returns
        edited_pitch, edited_periodicity, edited_loudness, edited_ppg
    """
    return from_features(
        torch.load(pitch_file),
        torch.load(periodicity_file),
        torch.load(loudness_file),
        torch.load(ppg_file),
        pitch_shift_cents,
        time_stretch_ratio,
        loudness_scale_db)


def from_file_to_file(
    pitch_file: Union[str, bytes, os.PathLike],
    periodicity_file: Union[str, bytes, os.PathLike],
    loudness_file: Union[str, bytes, os.PathLike],
    ppg_file: Union[str, bytes, os.PathLike],
    output_prefix: Union[str, bytes, os.PathLike],
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None
) -> None:
    """Edit speech representation on disk and save to disk

    Arguments
        pitch_file: Pitch file to edit
        periodicity_file: Periodicity file to edit
        loudness_file: Loudness file to edit
        ppg_file: PPG file to edit
        output_prefix: File to save output, minus extension
        pitch_shift_cents: Amount of pitch-shifting in cents
        time_stretch_ratio: Amount of time-stretching. Faster when above one.
        loudness_scale_db: Amount of loudness scaling in dB
    """
    # Edit
    pitch, periodicity, loudness, ppg = from_file(
        pitch_file,
        periodicity_file,
        loudness_file,
        ppg_file,
        pitch_shift_cents,
        time_stretch_ratio,
        loudness_scale_db)

    # Save
    torch.save(pitch, f'{output_prefix}-pitch.pt')
    torch.save(periodicity, f'{output_prefix}-periodicity.pt')
    torch.save(loudness, f'{output_prefix}-loudness.pt')
    torch.save(ppg, f'{output_prefix}-ppg.pt')


def from_files_to_files(
    pitch_files: List[Union[str, bytes, os.PathLike]],
    periodicity_files: List[Union[str, bytes, os.PathLike]],
    loudness_files: List[Union[str, bytes, os.PathLike]],
    ppg_files: List[Union[str, bytes, os.PathLike]],
    output_prefixes: List[Union[str, bytes, os.PathLike]],
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None
) -> None:
    """Edit speech representations on disk and save to disk

    Arguments
        pitch_files: Pitch files to edit
        periodicity_files: Periodicity files to edit
        loudness_files: Loudness files to edit
        ppg_files: PPG files to edit
        output_prefixes: Files to save output, minus extension
        pitch_shift_cents: Amount of pitch-shifting in cents
        time_stretch_ratio: Amount of time-stretching. Faster when above one.
        loudness_scale_db: Amount of loudness scaling in dB
    """
    for pitch_file, periodicity_file, loudness_file, ppg_file, prefix in zip(
        pitch_files,
        periodicity_files,
        loudness_files,
        ppg_files,
        output_prefixes
    ):
        from_file_to_file(
            pitch_file,
            periodicity_file,
            loudness_file,
            ppg_file,
            prefix,
            pitch_shift_cents,
            time_stretch_ratio,
            loudness_scale_db)
