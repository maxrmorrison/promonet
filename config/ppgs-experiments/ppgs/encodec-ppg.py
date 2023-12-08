from encodec import EncodecModel

MODULE = 'ppgs'

CONFIG = 'encodec-ppg'
INPUT_CHANNELS = 128 #dimensionality of encodec latents
REPRESENTATION = 'encodec'

LOCAL_CHECKPOINT = '/repos/ppgs/ppgs/assets/checkpoints/encodec.pt'

MAX_FRAMES = 10000

def _frontend(device='cpu'):
    import torch
    quantizer = EncodecModel.encodec_model_24khz().quantizer
    quantizer.to(device)

    def _quantize(batch: torch.Tensor):
        batch = batch.to(torch.int)
        batch = batch.transpose(0, 1)
        return quantizer.decode(batch)

    return _quantize

FRONTEND = _frontend