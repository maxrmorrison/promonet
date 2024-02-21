import torch

import promonet


###############################################################################
# Setup data loaders
###############################################################################


def loader(dataset, partition, adapt=promonet.ADAPTATION, gpu=None):
    """Setup data loader"""
    # Get dataset
    dataset = promonet.data.Dataset(dataset, partition, adapt)

    # Create loader
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=promonet.NUM_WORKERS,
        pin_memory=gpu is not None,
        collate_fn=promonet.data.collate,
        batch_sampler=promonet.data.sampler(dataset, partition))
