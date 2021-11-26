import promovits


###############################################################################
# Test preprocessing
###############################################################################


def test_mels(audio, true_mels):
    """Test melspectrogram preprocessing"""
    # Preprocess
    spectrogram = promovits.preprocess.spectrogram.from_audio(audio, mels=True)

    # Should be correct shape
    correct_shape = (promovits.NUM_MELS, audio.shape[1] // promovits.HOPSIZE)
    assert spectrogram.shape == correct_shape

    # Should be correct value
    assert (spectrogram == true_mels).all()


def test_spectrogram(audio, true_spectrogram):
    """Test spectrogram preprocessing"""
    # Preprocess
    spectrogram = promovits.preprocess.spectrogram.from_audio(audio)

    # Should be correct shape
    correct_shape = (
        promovits.NUM_FFT // 2 + 1,
        audio.shape[1] // promovits.HOPSIZE)
    assert spectrogram.shape == correct_shape

    # Should be correct value
    assert (spectrogram == true_spectrogram).all()


def test_text(text, true_features):
    """Test text preprocessing"""
    # Preprocess
    features = promovits.preprocess.text(text)

    # Should be correct shape
    correct_shape = (promovits.NUM_PHONEMES, len(text))
    assert features.shape == correct_shape

    # Should be correct value
    assert (features == true_features).all()
