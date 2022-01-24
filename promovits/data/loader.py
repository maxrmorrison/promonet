import torch

import promovits


###############################################################################
# Setup data loaders
###############################################################################


def loaders(dataset, train_partition, valid_partition, gpu=None):
    """Setup data loaders"""
    # Get dataset and collate function
    train_dataset = promovits.data.Dataset(dataset, train_partition)
    valid_dataset = promovits.data.Dataset(dataset, valid_partition)

    # Get sampler
    if torch.distributed.is_initialized():
        train_sampler = promovits.data.sampler.DistributedSampler(
            train_dataset,
            shuffle=True)
    else:
        train_sampler = promovits.data.sampler.Sampler(train_dataset)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=promovits.NUM_WORKERS,
        shuffle=False,
        pin_memory=gpu is not None,
        collate_fn=promovits.data.collate,
        batch_sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        num_workers=promovits.NUM_WORKERS,
        shuffle=False,
        batch_size=1,
        pin_memory=gpu is not None,
        drop_last=False,
        collate_fn=promovits.data.collate)

    return train_loader, valid_loader
