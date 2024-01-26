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
    stretch_silence: bool = True,
    return_grid: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
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
        return_grid: If true, also returns the time-stretch grid

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
                indices.append(ppgs.PHONEME_TO_INDEX_MAPPING[pypar.SILENCE])

            # Maybe add unvoiced
            if stretch_unvoiced:
                indices.extend(
                    list(
                        set(ppgs.PHONEMES) -
                        set(ppgs.VOICED) -
                        set([pypar.SILENCE])
                    )
                )

            # Get selection probabilities
            selected = ppg[torch.tensor(indices)].sum(dim=0)

            # Get number of output frames
            target_frames = round(ppg.shape[-1] / time_stretch_ratio)

            # Adjust ratio based on selection probabilities
            total_selected = selected.sum()
            total_unselected = ppg.shape[-1] - total_selected
            effective_ratio = (target_frames - total_unselected) / total_selected

            # TEMPORARY
            try:

                # Create time-stretch grid
                grid = torch.zeros(round(target_frames))
                i = 0.
                for j in range(1, target_frames):
                    left = math.floor(i)
                    offset = i - left
                    probability = (
                        offset * selected[left + 1] +
                        (1 - offset) * selected[left])
                    ratio = probability * effective_ratio + (1 - probability)
                    step = 1. / ratio
                    grid[j] = grid[j - 1] + step
                    i += step

            except IndexError as error:
                print(error)
                import pdb; pdb.set_trace()
                pass

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

    if return_grid:
        return pitch, periodicity, loudness, ppg, grid
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
    stretch_silence: bool = True,
    return_grid: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
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
        return_grid: If true, also returns the time-stretch grid

    Returns
        edited_pitch, edited_periodicity, edited_loudness, edited_ppg
    """
    pitch = torch.load(pitch_file)
    return from_features(
        pitch,
        torch.load(periodicity_file),
        torch.load(loudness_file),
        promonet.load.ppg(ppg_file, pitch.shape[-1]),
        pitch_shift_cents,
        time_stretch_ratio,
        loudness_scale_db,
        stretch_unvoiced,
        stretch_silence,
        return_grid)


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
    stretch_silence: bool = True,
    save_grid: bool = False
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
        save_grid: If true, also saves the time-stretch grid
    """
    # Edit
    results = from_file(
        pitch_file,
        periodicity_file,
        loudness_file,
        ppg_file,
        pitch_shift_cents,
        time_stretch_ratio,
        loudness_scale_db,
        stretch_unvoiced,
        stretch_silence,
        save_grid)

    # Save
    viterbi = '-viterbi' if promonet.VITERBI_DECODE_PITCH else ''
    torch.save(results[0], f'{output_prefix}{viterbi}-pitch.pt')
    torch.save(results[1], f'{output_prefix}{viterbi}-periodicity.pt')
    torch.save(results[2], f'{output_prefix}-loudness.pt')
    torch.save(results[3], f'{output_prefix}{ppgs.representation_file_extension()}')
    if save_grid:
        torch.save(results[4], f'{output_prefix}-grid.pt')


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
    stretch_silence: bool = True,
    save_grid: bool = False
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
        save_grid: If true, also saves the time-stretch grid
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
            stretch_silence,
            save_grid)
