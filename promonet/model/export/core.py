import torchutil

import promonet


###############################################################################
# Model exporting
###############################################################################


def from_file_to_file(checkpoint=None, output_file='promonet-export.ts'):
    """Load model from checkpoint and export to torchscript"""
    # Load model
    model = promonet.model.Generator()
    if checkpoint is not None:
        model, *_ = torchutil.checkpoint.load(checkpoint, model)

    # Switch to evaluation mode
    with torchutil.inference.context(model):

        # Export
        model.export(output_file)
