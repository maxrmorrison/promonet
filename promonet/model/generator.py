import ppgs
import torch

import promonet


###############################################################################
# Base generator definition
###############################################################################


class BaseGenerator(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Model selection
        if promonet.MODEL == 'fargan':
            self.model = promonet.model.FARGAN(
                promonet.NUM_FEATURES,
                promonet.GLOBAL_CHANNELS)
        elif promonet.MODEL == 'hifigan':
            self.model = promonet.model.HiFiGAN(
                promonet.NUM_FEATURES,
                promonet.GLOBAL_CHANNELS)
        elif promonet.MODEL == 'vocos':
            self.model = promonet.model.Vocos(
                promonet.NUM_FEATURES,
                promonet.GLOBAL_CHANNELS)
        else:
            raise ValueError(
                f'Generator model {promonet.MODEL} is not defined')

        # Speaker embedding
        if promonet.ZERO_SHOT:
            self.speaker_embedding = torch.nn.Linear(
                promonet.WAVLM_EMBEDDING_CHANNELS,
                promonet.SPEAKER_CHANNELS)
        else:
            self.speaker_embedding = torch.nn.Embedding(
                promonet.NUM_SPEAKERS,
                promonet.SPEAKER_CHANNELS)

        # Default value for previous samples
        self.register_buffer(
            'default_previous_samples',
            torch.zeros(1, 1, promonet.NUM_PREVIOUS_SAMPLES))

    def prepare_global_features(
        self,
        speakers,
        spectral_balance_ratios,
        loudness_ratios
    ):
        # Encode speaker
        global_features = self.speaker_embedding(speakers).unsqueeze(-1)

        # Maybe add augmentation ratios
        if promonet.AUGMENT_PITCH:
            global_features = torch.cat(
                (global_features, spectral_balance_ratios[:, None, None]),
                dim=1)

        # Maybe add augmentation ratios
        if promonet.AUGMENT_LOUDNESS:
            global_features = torch.cat(
                (global_features, loudness_ratios[:, None, None]),
                dim=1)

        return global_features

    def remove_weight_norm(self):
        """Remove weight normalization for scriptable inference"""
        try:
            self.model.remove_weight_norm()
        except AttributeError:
            pass


###############################################################################
# Proposed generator definition
###############################################################################


class Generator(BaseGenerator):

    def __init__(self):
        super().__init__()

        # Pitch embedding
        if 'pitch' in promonet.INPUT_FEATURES and promonet.PITCH_EMBEDDING:
            self.pitch_embedding = torch.nn.Embedding(
                promonet.PITCH_BINS,
                promonet.PITCH_EMBEDDING_SIZE)

        # PPG sparsity threshold
        if (
            promonet.SPARSE_PPG_METHOD is not None and
            ppgs.REPRESENTATION_KIND == 'ppg'
        ):
            ppg_threshold = torch.tensor(
                promonet.SPARSE_PPG_THRESHOLD,
                dtype=torch.float)
            self.register_buffer('ppg_threshold', ppg_threshold)

        # Torchscript prohibits Python lists in forward
        self.use_pitch = 'pitch' in promonet.INPUT_FEATURES
        self.use_loudness = 'loudness' in promonet.INPUT_FEATURES
        self.use_periodicity = 'periodicity' in promonet.INPUT_FEATURES

        # Torchscript prohibits try/except or attribute lookup in forward
        if self.use_pitch and promonet.VARIABLE_PITCH_BINS:
            pitch_distribution = promonet.load.pitch_distribution()
            self.register_buffer('pitch_distribution', pitch_distribution)

    def forward(
        self,
        loudness,
        pitch,
        periodicity,
        ppg,
        speakers,
        spectral_balance_ratios,
        loudness_ratios,
        previous_samples
    ):
        # Prepare input features
        features = self.prepare_features(loudness, pitch, periodicity, ppg)
        global_features = self.prepare_global_features(
            speakers,
            spectral_balance_ratios,
            loudness_ratios)

        # Synthesize
        return self.model(features, global_features, previous_samples)

    def prepare_features(self, loudness, pitch, periodicity, ppg):
        """Prepare input features for training or inference"""
        # Maybe sparsify PPGs
        if (
            promonet.SPARSE_PPG_METHOD is not None and
            ppgs.REPRESENTATION_KIND == 'ppg'
        ):
            ppg = ppgs.sparsify(
                ppg,
                promonet.SPARSE_PPG_METHOD,
                self.ppg_threshold)

        features = ppg

        # Maybe add pitch features
        if self.use_pitch:
            hz = torch.clip(pitch, promonet.FMIN, promonet.FMAX)
            if promonet.PITCH_EMBEDDING:
                if promonet.VARIABLE_PITCH_BINS:
                    bins = torch.searchsorted(self.pitch_distribution, hz)
                    bins = torch.clip(bins, 0, promonet.PITCH_BINS - 1)
                else:
                    normalized = (
                        (torch.log2(hz) - promonet.LOG_FMIN) /
                        (promonet.LOG_FMAX - promonet.LOG_FMIN))
                    bins = (
                        (promonet.PITCH_BINS - 1) * normalized).to(torch.long)
                pitch_embeddings = self.pitch_embedding(bins).permute(0, 2, 1)
            else:
                pitch_embeddings = (
                    (torch.log2(hz)[:, None] - promonet.LOG_FMIN) /
                    (promonet.LOG_FMAX - promonet.LOG_FMIN))
            features = torch.cat((features, pitch_embeddings), dim=1)

        # Maybe add loudness features
        if self.use_loudness:
            bands = promonet.LOUDNESS_BANDS
            step = loudness.shape[-2] / bands
            averaged = torch.stack(
                [
                    loudness[:, int(band * step):int((band + 1) * step)].mean(dim=-2)
                    for band in range(bands)
                ],
                dim=1)
            normalized = promonet.preprocess.loudness.normalize(averaged)
            if normalized.ndim == 2:
                normalized = normalized[None]
            features = torch.cat((features, normalized), dim=1)

        # Maybe add periodicity features
        if self.use_periodicity:
            features = torch.cat((features, periodicity[:, None]), dim=1)

        # Append period for FARGAN pitch lookup
        if promonet.MODEL == 'fargan':
            period = (
                promonet.SAMPLE_RATE /
                torch.clip(pitch, promonet.FMIN, promonet.FMAX))
            features = torch.cat((features, period[:, None]), dim=1)

        return features

    ###########################################################################
    # Model exporting
    ###########################################################################

    def export(self, output_file):
        """Export model using torchscript"""
        # Remove weight normalization
        self.remove_weight_norm()

        # Register packed inference method
        self.register()

        # Run torchscript
        scripted = torch.jit.script(self)

        # Save
        scripted.save(output_file)

    @torch.jit.export
    def get_attributes(self):
        return ['none']

    @torch.jit.export
    def get_methods(self):
        return self._methods

    def labels(self):
        """Retrieve labels for input channels"""
        labels = []

        # Loudness
        labels += [
            f'loudness-{i}' for i in range(promonet.LOUDNESS_BANDS)]

        # Pitch
        labels.append('pitch')

        # Periodicity
        labels.append('periodicity')

        # PPG
        labels += [
            f'ppg-{i} ({ppgs.PHONEMES[i]})'
            for i in range(promonet.PPG_CHANNELS)]

        # Speaker
        labels.append('speaker')

        # Spectral balance
        labels.append('spectral balance')

        # Loudness ratio
        labels.append('loudness ratio')

        return labels

    def pack_features(
        self,
        loudness,
        pitch,
        periodicity,
        ppg,
        speakers,
        spectral_balance_ratios,
        loudness_ratios
    ):
        """Pack features into a single frame-resolution tensor"""
        features = torch.zeros((loudness.shape[0], 0, loudness.shape[2]))

        # Loudness
        if self.use_loudness:
            averaged = promonet.preprocess.loudness.band_average(loudness)
            features = torch.cat((features, averaged), dim=1)

        # Pitch
        if self.use_pitch:
            features = torch.cat((features, pitch), dim=1)

        # Periodicity
        if self.use_periodicity:
            features = torch.cat((features, periodicity), dim=1)

        # PPG
        if (
            promonet.SPARSE_PPG_METHOD is not None and
            ppgs.REPRESENTATION_KIND == 'ppg'
        ):
            ppg = ppgs.sparsify(
                ppg,
                promonet.SPARSE_PPG_METHOD,
                self.ppg_threshold)
        features = torch.cat((features, ppg), dim=1)

        # Speaker
        speakers = speakers[:, None, None].repeat(1, 1, features.shape[-1])
        features = torch.cat((features, speakers.to(torch.float)), dim=1)

        # Spectral balance
        if promonet.AUGMENT_PITCH:
            spectral_balance_ratios = \
                spectral_balance_ratios[:, None, None].repeat(
                    1, 1, features.shape[-1])
            features = torch.cat((features, spectral_balance_ratios), dim=1)

        # Loudness ratio
        if promonet.AUGMENT_LOUDNESS:
            loudness_ratios = loudness_ratios[:, None, None].repeat(
                1, 1, features.shape[-1])
            features = torch.cat((features, loudness_ratios), dim=1)

        return features

    @torch.jit.export
    def packed_inference(self, x):
        """Export function

        Arguments
            x
                Frame-resolution input features

        Returns
            audio
                Synthesized speech
        """
        (
            loudness,
            pitch,
            periodicity,
            ppg,
            speakers,
            spectral_balance_ratios,
            loudness_ratios
        ) = self.unpack_features(x)

        # torch.jit does not support keyword argument unpacking
        return self(
            loudness,
            pitch,
            periodicity,
            ppg,
            speakers,
            spectral_balance_ratios,
            loudness_ratios,
            self.default_previous_samples
        ).to(torch.float)

    def register(
        self,
        method_name: str = 'packed_inference',
        test_buffer_size: int = 8192
    ):
        """Register a class method for use by IRCAM's nn~"""
        # Get semantic labels for each input channel
        labels = self.labels()

        # Create buffer that stores input/output sizes
        self.register_buffer(
            f'{method_name}_params',
            torch.tensor([len(labels), promonet.HOPSIZE, 1, 1]))

        # Label each input/output channel
        setattr(self, f'{method_name}_input_labels', labels)
        setattr(self, f'{method_name}_output_labels', ['output audio'])

        # Test packed inference
        x = torch.zeros(1, len(labels), test_buffer_size // promonet.HOPSIZE)
        y = getattr(self, method_name)(x)
        assert (
            tuple(y.shape) == (1, 1, test_buffer_size) and
            y.dtype == torch.float)

        # Register packed inference method
        self._methods = [method_name]

    def unpack_features(self, x):
        """Unpack frame-resolution features

        Features:
            loudness
            pitch
            periodicity
            ppg
            speaker
            spectral balance
            loudness ratio
        """
        i = 0

        # Loudness
        loudness = x[:, i:i + promonet.LOUDNESS_BANDS]
        i += promonet.LOUDNESS_BANDS

        # Pitch
        pitch = x[:, i:i + 1].squeeze(1)
        i += 1

        # Periodicity
        periodicity = x[:, i:i + 1].squeeze(1)
        i += 1

        # PPG
        ppg = x[:, i:i + promonet.PPG_CHANNELS]
        i += promonet.PPG_CHANNELS

        # Speaker
        speakers = x[:, i:i + 1, 0].to(torch.long).squeeze(1)
        i += 1

        # Spectral balance
        spectral_balance_ratios = x[:, i:i + 1, 0].squeeze(1)
        i += 1

        # Loudness ratio
        loudness_ratios = x[:, i:i + 1, 0].squeeze(1)
        i += 1

        return (
            loudness,
            pitch,
            periodicity,
            ppg,
            speakers,
            spectral_balance_ratios,
            loudness_ratios)


###############################################################################
# Mel vocoder
###############################################################################


class MelGenerator(BaseGenerator):
    """Generate speech from Mel spectrograms"""

    def forward(
        self,
        spectrograms,
        speakers,
        spectral_balance_ratios,
        loudness_ratios,
        previous_samples
    ):
        # Prepare input features
        features = self.prepare_features(spectrograms)
        global_features = self.prepare_global_features(
            speakers,
            spectral_balance_ratios,
            loudness_ratios)

        # Synthesize
        if promonet.MODEL == 'cargan':
            return self.model(features, global_features, previous_samples)
        return self.model(features, global_features)

    def prepare_features(self, spectrograms):
        """Prepare input features for training or inference"""
        # Convert linear spectrogram to Mels
        features = promonet.preprocess.spectrogram.linear_to_mel(
            spectrograms)

        # Sparsify by adding the clipping threshold
        if promonet.SPARSE_MELS:
            features += promonet.LOG_DYNAMIC_RANGE_COMPRESSION_THRESHOLD

        return features
