import torch

import promonet


###############################################################################
# Setup data loaders
###############################################################################


def loader(datasets, partition, adapt=promonet.ADAPTATION, gpu=None):
    """Setup data loader"""
    if len(datasets) == 1:
        # Get dataset
        dataset = datasets[0]
        dataset = promonet.data.Dataset(dataset, partition, adapt)
    else:
        # Use a ConcatDataset
        dataset = promonet.data.ConcatDataset(
            datasets = [promonet.data.Dataset(dataset, partition, adapt) for dataset in datasets]
        )
        
    # Create loader
    return torch.utils.data.DataLoader(
        dataset,
        num_workers   = promonet.NUM_WORKERS,
        pin_memory    = gpu is not None,
        collate_fn    = promonet.data.collate,
        batch_sampler = promonet.data.sampler(dataset, partition)
    )