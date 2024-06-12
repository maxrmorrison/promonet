import torchutil

import promonet


###############################################################################
# Model exporting
###############################################################################


def from_file_to_file(checkpoint=None, output_file='promonet-export.ts'):
    """Load model from checkpoint and export to torchscript"""
    model = promonet.model.Generator()
    if checkpoint is not None:
        model, *_ = torchutil.checkpoint.load(checkpoint, model)
    model.export(output_file)
