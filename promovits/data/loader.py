
import torch

import promovits


###############################################################################
# Setup data loaders
###############################################################################


def loaders(rank, world_size, dataset, gpu=None, use_ppg=False, interp_method='nearest'):
    """Setup data loaders"""
    # TODO - File lists
    # TODO - Resolve dataset name
    # TODO - data hyperparameters
    # TODO - get rank and world_size from torch.distributed

    # Get training dataset and collate function
    if use_ppg:
        train_dataset = promovits.data.datasets.PPGAudioSpeakerLoader(
            train_files,
            hps.data)
        collate_fn = promovits.data.collate.PPGAudioSpeakerCollate()
    else:
        train_dataset = promovits.data.datasets.TextAudioSpeakerLoader(
            train_files,
            hps.data)
        collate_fn = promovits.data.collate.TextAudioSpeakerCollate()

    # Get sampler
    boundaries = [32,300,400,500,600,700,800,900,1000]
    if rank is None:
        train_sampler = promovits.data.sampler.RandomBucketSampler(
            train_dataset,
            promovits.BATCH_SIZE,
            boundaries)
    else:
        train_sampler = promovits.data.sampler.DistributedBucketSampler(
            train_dataset,
            promovits.BATCH_SIZE,
            boundaries,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    # Make training loader
    train_loader = torch.data.utils.DataLoader(
        train_dataset,
        num_workers=promovits.NUM_WORKERS,
        shuffle=False,
        pin_memory=gpu is not None,
        collate_fn=collate_fn,
        batch_sampler=train_sampler)

    if not rank:

        # Get evaluation dataset
        if use_ppg:
            eval_dataset = promovits.data.datasets.PPGAudioSpeakerLoader(
                hps.data.validation_files,
                hps.data)
        else:
            eval_dataset = promovits.data.datasets.TextAudioSpeakerLoader(
                hps.data.validation_files,
                hps.data)

        # Make evaluation loader
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            num_workers=promovits.NUM_WORKERS,
            shuffle=False,
            batch_size=promovits.BATCH_SIZE,
            pin_memory=gpu is not None,
            drop_last=False,
            collate_fn=collate_fn)

    return train_loader, eval_loader
