import math
import os
from typing import List, Optional, Tuple, Union

import ppgs
import pypar
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
    loudness_scale_db: Optional[float] = None,
    stretch_unvoiced: bool = True,
    stretch_silence: bool = True
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
        stretch_unvoiced: If true, applies time-stretching to unvoiced frames
        stretch_silence: If true, applies time-stretching to silent frames

    Returns
        edited_pitch, edited_periodicity, edited_loudness, edited_ppg
    """
    # Maybe time-stretch
    if time_stretch_ratio is not None:

        # Create time-stretch grid
        if stretch_unvoiced and stretch_silence:
            grid = promonet.edit.grid.constant(
                ppg,
                time_stretch_ratio)
        else:

            # Get voiced phoneme indices
            indices = [
                ppgs.PHONEME_TO_INDEX_MAPPING[phoneme]
                for phoneme in ppgs.VOICED]

            # Maybe add silence
            if stretch_silence:
                indices.append(pypar.SILENCE)

            # Maybe add unvoiced
            if stretch_unvoiced:
                indices.extend(
                    list(
                        set(ppgs.PHONEMES) -
                        set(ppgs.VOICED) -
                        set([pypar.SILENCE])
                    )
                )

            # Get frames where sum of selected probabilities exceeds threshold
            selected = ppg[torch.tensor(indices)].sum(dim=0) > .5

            # Compute effective ratio on selected frames
            target_frames = math.round(time_stretch_ratio * ppg.shape[-1])
            num_selected = selected.sum()
            effective_ratio = (
                target_frames - ppg.shape[-1] + num_selected) / num_selected

            # Create time-stretch grid
            grid = torch.zeros(target_frames)
            i = 0.
            for j in range(1, target_frames):
                idx = int(round(i))
                step = effective_ratio.item() if selected[idx] else 1.
                grid[j] = grid[j - 1] + step
                i += step
            grid[-1] = torch.clip(grid[-1], max=len(ppg) - 1)

        # Time-stretch
        pitch = 2 ** promonet.edit.grid.sample(torch.log2(pitch), grid)
        periodicity = promonet.edit.grid.sample(periodicity, grid)
        loudness = promonet.edit.grid.sample(loudness, grid)
        ppg = promonet.edit.grid.sample(ppg, grid, promonet.PPG_INTERP_METHOD)

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
    loudness_scale_db: Optional[float] = None,
    stretch_unvoiced: bool = True,
    stretch_silence: bool = True
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
        stretch_unvoiced: If true, applies time-stretching to unvoiced frames
        stretch_silence: If true, applies time-stretching to silent frames

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
        loudness_scale_db,
        stretch_unvoiced,
        stretch_silence)


def from_file_to_file(
    pitch_file: Union[str, bytes, os.PathLike],
    periodicity_file: Union[str, bytes, os.PathLike],
    loudness_file: Union[str, bytes, os.PathLike],
    ppg_file: Union[str, bytes, os.PathLike],
    output_prefix: Union[str, bytes, os.PathLike],
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None,
    stretch_unvoiced: bool = True,
    stretch_silence: bool = True
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
        stretch_unvoiced: If true, applies time-stretching to unvoiced frames
        stretch_silence: If true, applies time-stretching to silent frames
    """
    # Edit
    pitch, periodicity, loudness, ppg = from_file(
        pitch_file,
        periodicity_file,
        loudness_file,
        ppg_file,
        pitch_shift_cents,
        time_stretch_ratio,
        loudness_scale_db,
        stretch_unvoiced,
        stretch_silence)

    # Save
    torch.save(pitch, f'{output_prefix}-pitch.pt')
    torch.save(periodicity, f'{output_prefix}-periodicity.pt')
    torch.save(loudness, f'{output_prefix}-loudness.pt')
    torch.save(ppg, f'{output_prefix}{ppgs.representation_file_extension()}')


def from_files_to_files(
    pitch_files: List[Union[str, bytes, os.PathLike]],
    periodicity_files: List[Union[str, bytes, os.PathLike]],
    loudness_files: List[Union[str, bytes, os.PathLike]],
    ppg_files: List[Union[str, bytes, os.PathLike]],
    output_prefixes: List[Union[str, bytes, os.PathLike]],
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None,
    stretch_unvoiced: bool = True,
    stretch_silence: bool = True
) -> None:
    """Edit speech representations on disk and save to disk

    Arguments
        pitch_files: Pitch files to edit
        periodicity_files: Periodicity files to edit
        loudness_files: Loudness files to edit
        ppg_files: Phonetic posteriorgram files to edit
        output_prefixes: Files to save output, minus extension
        pitch_shift_cents: Amount of pitch-shifting in cents
        time_stretch_ratio: Amount of time-stretching. Faster when above one.
        loudness_scale_db: Amount of loudness scaling in dB
        stretch_unvoiced: If true, applies time-stretching to unvoiced frames
        stretch_silence: If true, applies time-stretching to silent frames
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
            loudness_scale_db,
            stretch_unvoiced,
            stretch_silence)
