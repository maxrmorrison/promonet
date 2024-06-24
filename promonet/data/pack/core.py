import ppgs
import torch

import csv
import numpy as np

import promonet


###############################################################################
# Pack features
###############################################################################


def from_audio(audio, speaker=0, spectral_balance_ratio=1., gpu=None):
    """Convert audio to packed features"""
    # Preprocess audio
    loudness, pitch, periodicity, ppg = promonet.preprocess.from_audio(
        audio,
        gpu=gpu)

    # Pack features
    return from_features(
        loudness[None].cpu(),
        pitch[None].cpu(),
        periodicity[None].cpu(),
        ppg.cpu(),
        speaker,
        spectral_balance_ratio,
        1.)


def from_features(
    loudness,
    pitch,
    periodicity,
    ppg,
    speaker=0,
    spectral_balance_ratio=1.,
    loudness_ratio=1.):
    """Pack features into a single tensor"""
    features = torch.zeros((loudness.shape[0], 0, loudness.shape[2]))

    # Loudness
    averaged = promonet.preprocess.loudness.band_average(loudness)
    features = torch.cat((features, averaged), dim=1)

    # Pitch
    features = torch.cat((features, pitch), dim=1)

    # Periodicity
    features = torch.cat((features, periodicity), dim=1)

    # PPG
    if (
        promonet.SPARSE_PPG_METHOD is not None and
        ppgs.REPRESENTATION_KIND == 'ppg'
    ):
        threshold = torch.tensor(
            promonet.SPARSE_PPG_THRESHOLD,
            dtype=torch.float)
        ppg = ppgs.sparsify(ppg, promonet.SPARSE_PPG_METHOD, threshold)
    features = torch.cat((features, ppg), dim=1)

    # Speaker
    speaker = torch.tensor([speaker])[:, None, None]
    speaker = speaker.repeat(1, 1, features.shape[-1]).to(torch.float)
    features = torch.cat((features, speaker), dim=1)

    # Spectral balance
    spectral_balance_ratio = \
        torch.tensor([spectral_balance_ratio])[:, None, None].repeat(
            1, 1, features.shape[-1])
    features = torch.cat((features, spectral_balance_ratio), dim=1)

    # Loudness ratio
    loudness_ratio = \
        torch.tensor([loudness_ratio])[:, None, None].repeat(
        1, 1, features.shape[-1])
    features = torch.cat((features, loudness_ratio), dim=1)

    return features


def from_file_to_file(
    audio_file,
    output_file = None,
    speaker = 0,
    spectral_balance_ratio=1.,
    gpu=None):
    """Convert audio file to packed features and save"""
    # Default to audio_file with .csv extension
    if output_file is None:
        output_format = 'csv'
    else:
        output_format = output_file.suffix[1:]
        if output_format not in ['csv', 'pt']:
            raise ValueError(f'Output Format "{output_format}" is not supported')
    output_file = audio_file.with_suffix(f'.{output_format}')
    
    # Pack features
    audio = promonet.load.audio(audio_file)
    features = from_audio(
        audio,
        speaker,
        spectral_balance_ratio,
        gpu)
    
    # Save
    if output_format == 'pt':
        torch.save(features, output_file)
    elif output_format == 'csv':
        features = features.cpu().numpy()[0]
        
        # Representation labels for header
        labels = [
            *[f'loudness-{i}' for i in range(promonet.LOUDNESS_BANDS)],                 # Loudness (8)
            'pitch',                                                                    # Pitch
            'periodicity',                                                              # Periodicity
            *[f'ppg-{i} ({ppgs.PHONEMES[i]})' for i in range(promonet.PPG_CHANNELS)],   # PPG (40)
            'speaker',                                                                  # Speaker id
            'spectral balance',                                                         # Spectral Balance
            'loudness ratio'                                                            # Loudness Ratio
        ]
        labels = ['timecode', *labels]                                                  # Start of frame time (seconds)
        
        # Generate timecode information (frame beginning)
        timecodes = np.arange(0.0, audio.shape[-1] / promonet.SAMPLE_RATE, promonet.HOPSIZE / promonet.SAMPLE_RATE)
        timecodes = timecodes[:features.shape[-1]]
        
        # Save to CSV
        with open(output_file, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(labels)
            for i in range(features.shape[-1]):
                row = [timecodes[i], *features[:,i].tolist()]
                row = [(f"{int(r)}" if i == 51 else f"{r:.8f}") for i,r in enumerate(row)]
                csv_writer.writerow(row)