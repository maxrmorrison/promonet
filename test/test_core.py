import promonet


###############################################################################
# Test generation
###############################################################################


def test_core(audio):
    """Test generation"""
    # Preprocess
    features = promonet.preprocess.spectrogram.from_audio(audio)

    # Vocode
    vocoded_from_features = promonet.from_features(features)
    vocoded_from_audio = promonet.from_audio(audio, promonet.SAMPLE_RATE)

    # Should be deterministic
    assert (vocoded_from_audio == vocoded_from_features).all()

    # Should be correct length
    assert vocoded_from_audio.shape[-1] == features.shape[-1] * promonet.HOPSIZE
