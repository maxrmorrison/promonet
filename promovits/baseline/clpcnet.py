import clpcnet
import numpy as np
import torch

import promovits


def from_audio(
    audio,
    sample_rate=promovits.SAMPLE_RATE,
    text=None,
    grid=None,
    target_loudness=None,
    target_pitch=None,
    checkpoint=promovits.DEFAULT_CHECKPOINT,
    gpu=None):
    """Speech modification with CLPCNet"""
    if grid and not text:
        raise ValueError('Cannot use time-stretch grid without text')

    # Resample audio
    audio = clpcnet.preprocess.resample(audio.squeeze().numpy(), sample_rate)

    # Require a minimum peak amplitude
    maximum = np.abs(audio).max()
    if maximum < .2:
        audio = audio / maximum * .4

    # Compute BFCCs
    features = clpcnet.preprocess.from_audio(audio)

    # Compute pitch and periodicity
    pitch, periodicity = clpcnet.pitch.from_audio(audio, gpu)

    # Scale crepe periodicity to match range of yin
    periodicity = .8 * periodicity - .4

    # Replace periodicity
    features[0, :, clpcnet.CORRELATION_IDX] = periodicity

    # Maybe pitch-shift
    pitch = pitch if target_pitch is None else target_pitch.squeeze().numpy()

    # Bound pitch
    pitch[pitch < clpcnet.FMIN] = clpcnet.FMIN
    pitch[pitch > clpcnet.FMAX] = clpcnet.FMAX

    # Replace pitch
    features[:, :, clpcnet.PITCH_IDX] = clpcnet.convert.hz_to_epochs(
        pitch)[None]

    # Convert pitch to bin indices
    pitch_bins = clpcnet.convert.hz_to_bins(pitch).reshape(1, -1, 1)

    # Maybe time-stretch
    if grid:

        # Get total number of samples to generate
        samples = clpcnet.HOPSIZE * len(grid)

        # Normalize to [0, 1]
        norm = grid / features.shape[1]

        # Get cumulative sum of number of samples per frame
        cumsum = norm * samples

        # Get hopsize per frame
        diff = cumsum[1:] - cumsum[:-1]

        # Recover lost frame
        diff = torch.nn.functional.interpolate(
            diff.to(torch.float)[None, None],
            len(diff) + 1,
            mode='linear')

        # Round to exactly the number of samples to generate
        hopsizes = diff.to(torch.long)
        residuals = diff - hopsizes.to(torch.float)
        compensation = samples - hopsizes.sum()
        indices = torch.topk(residuals, compensation)
        hopsizes[indices] += 1

    else:

        hopsizes = [clpcnet.HOPSIZE] * features.shape[1]

    # Maybe scale loudness
    if target_loudness:

        # TODO
        pass

    # Create session and pre-load model
    if not hasattr(clpcnet.from_features, 'session') or \
       (clpcnet.from_features.session.gpu != gpu) or \
       (checkpoint is not None and
            clpcnet.from_features.session.file != checkpoint):
        clpcnet.load.model(checkpoint, gpu)

    # Setup tensorflow session
    with clpcnet.from_features.session.context():

        # Run frame-rate network
        frame_rate_feats = from_features.session.encoder.predict(
            [features[:, :, :clpcnet.SPECTRAL_FEATURE_SIZE], pitch_bins])

        # Run sample-rate network
        generated = clpcnet.core.decode(
            features,
            frame_rate_feats,
            hopsizes,
            False)

    # Scale to original peak amplitude if necessary
    if maximum < .2:
        generated = generated / .4 * maximum

    return generated
