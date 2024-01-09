import math

import torch

import promonet


###############################################################################
# Model definition
###############################################################################


class Generator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Model selection
        if promonet.MODEL == 'hifigan':
            self.vocoder = promonet.model.HiFiGAN(
                promonet.NUM_FEATURES,
                promonet.GLOBAL_CHANNELS)
        elif promonet.MODEL == 'vits':
            self.synthesizer = promonet.model.VITS()
            self.vocoder = promonet.model.HiFiGAN(
                promonet.VITS_CHANNELS,
                promonet.GLOBAL_CHANNELS)
        elif promonet.MODEL == 'vocos':
            self.vocoder = promonet.model.HiFiGAN(
                promonet.NUM_FEATURES,
                promonet.GLOBAL_CHANNELS)

        # Speaker embedding
        self.speaker_embedding = torch.nn.Embedding(
            promonet.NUM_SPEAKERS,
            promonet.SPEAKER_CHANNELS)

        # Pitch embedding
        if 'pitch' in promonet.INPUT_FEATURES and promonet.PITCH_EMBEDDING:
            self.pitch_embedding = torch.nn.Embedding(
                promonet.PITCH_BINS,
                promonet.PITCH_EMBEDDING_SIZE)

    def forward(
        self,
        ppgs,
        pitch,
        periodicity,
        loudness,
        lengths,
        speakers,
        formant_ratios=None,
        loudness_ratios=None,
        spectrograms=None
    ):
        # Prepare input features
        features, global_features = self.prepare_features(
            ppgs,
            pitch,
            periodicity,
            loudness,
            speakers,
            formant_ratios,
            loudness_ratios,
            spectrograms)

        # Maybe encode representation to match spectrogram
        if promonet.MODEL == 'vits':
            features, *vits_args = self.synthesizer(
                features,
                spectrograms,
                lengths,
                global_features)
        else:
            vits_args = None

        # Decode representation to waveform
        generated = self.vocoder(features, global_features)

        return generated, vits_args

    def prepare_features(
        self,
        ppgs,
        pitch,
        periodicity,
        loudness,
        speakers,
        formant_ratios,
        loudness_ratios,
        spectrograms
    ):
        """Prepare input features for training or inference"""
        # Standard vocoding from Mel spectrograms
        if promonet.SPECTROGRAM_ONLY:
            mels = promonet.preprocess.spectrogram.linear_to_mel(
                spectrograms)
            return mels, None

        features = ppgs

        # Maybe add pitch features
        if 'pitch' in promonet.INPUT_FEATURES:
            if promonet.PITCH_EMBEDDING:
                pitch = promonet.convert.hz_to_bins(pitch)
                pitch_embeddings = self.pitch_embedding(pitch).permute(0, 2, 1)
            else:
                pitch_embeddings = (
                    (torch.log2(pitch)[:, None] - promonet.LOG_FMIN) /
                    (promonet.LOG_FMAX - promonet.LOG_FMIN))
            features = torch.cat((features, pitch_embeddings), dim=1)

        # Maybe add loudness features
        if 'loudness' in promonet.INPUT_FEATURES:
            normalized = promonet.loudness.normalize(loudness)
            features = torch.cat((features, normalized[:, None]), dim=1)

        # Maybe add periodicity features
        if 'periodicity' in promonet.INPUT_FEATURES:
            features = torch.cat((features, periodicity[:, None]), dim=1)

        # Default augmentation ratio is 1
        if formant_ratios is None and promonet.AUGMENT_PITCH:
            formant_ratios = torch.ones(
                1 if ppgs.dim() == 2 else len(ppgs),
                dtype=torch.float,
                device=ppgs.device)
        if loudness_ratios is None and promonet.AUGMENT_LOUDNESS:
            loudness_ratios = torch.ones(
                1 if ppgs.dim() == 2 else len(ppgs),
                dtype=torch.float,
                device=ppgs.device)

        # Encode speaker ID
        global_features = self.speaker_embedding(speakers).unsqueeze(-1)

        # Maybe add augmentation ratios
        if ('pitch' in promonet.INPUT_FEATURES) and promonet.AUGMENT_PITCH:
            global_features = torch.cat(
                (global_features, formant_ratios[:, None, None]),
                dim=1)

        # Maybe add augmentation ratios
        if ('loudness' in promonet.INPUT_FEATURES) and promonet.AUGMENT_LOUDNESS:
            global_features = torch.cat(
                (global_features, loudness_ratios[:, None, None]),
                dim=1)

        return features, global_features
