from encodec import EncodecModel

MODULE = 'ppgs'

CONFIG = 'encodec-latents'

REPRESENTATION = 'encodec'

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