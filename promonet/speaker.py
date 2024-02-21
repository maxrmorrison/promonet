import torch
import torchaudio

import promonet


###############################################################################
# Wespeaker speaker embedding
###############################################################################


def from_audio(audio, gpu=None):
    """Embed audio"""
    # Save audio to temporary storage
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        file = directory / 'speaker-embedding.wav'
        torchaudio.save(file, audio, promonet.SAMPLE_RATE)

        # Embed
        return from_file(file, gpu)


def from_file(file, gpu=None):
    """Embed audio on disk"""
    # Cache model
    if not hasattr(from_file, 'model'):
        import wespeaker
        from_file.model = wespeaker.load_model('english')

    # Maybe move model
    if not hasattr(from_file, 'gpu') or from_file.gpu != gpu:
        from_file.model.set_gpu(gpu)
        from_file.gpu = gpu

    # Embed
    return from_file.model.extract_embedding(file)


def from_file_to_file(input_file, output_file, gpu=None):
    """Embed audio on disk and save"""
    # Embed
    embedding = from_file(input_file, gpu).cpu()

    # Save
    torch.save(embedding, output_file)


def from_files(files, gpu=None):
    """Embed audio files on disk to a single embedding"""
    # Load and concatenate audio
    audio = torch.cat([promonet.load.audio(file) for file in files], dim=1)

    # Embed
    return from_audio(audio, gpu)


def from_files_to_file(input_files, output_file, gpu=None):
    """Embed audio files to a single embedding and save"""
    # Embed
    embedding = from_files(input_files, gpu).cpu()

    # Save
    torch.save(embedding, output_file)


def from_files_to_files(input_files, output_files, gpu=None):
    """Embed audio files to independent embeddings and save"""
    for input_file, output_file in zip(input_files, output_files):
        from_file_to_file(input_file, output_file, gpu)
