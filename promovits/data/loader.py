
import torch

import promovits


###############################################################################
# Setup data loaders
###############################################################################


def loaders(dataset, gpu=None, use_ppg=False, interp_method='nearest'):
    """Setup data loaders"""
    # Get dataset and collate function
    if use_ppg:
        train_dataset = promovits.data.datasets.PPGAudioSpeakerLoader(
            dataset,
            'train',
            interp_method)
        valid_dataset = promovits.data.datasets.PPGAudioSpeakerLoader(
            dataset,
            'valid',
            interp_method)
        collate_fn = promovits.data.collate.PPGAudioSpeakerCollate()
    else:
        train_dataset = promovits.data.datasets.TextAudioSpeakerLoader(
            dataset,
            'train')
        valid_dataset = promovits.data.datasets.TextAudioSpeakerLoader(
            dataset,
            'valid')
        collate_fn = promovits.data.collate.TextAudioSpeakerCollate()

    # Get sampler
    boundaries = [32,300,400,500,600,700,800,900,1000]
    if torch.distributed.is_initialized():
        train_sampler = promovits.data.sampler.DistributedBucketSampler(
            train_dataset,
            promovits.BATCH_SIZE,
            boundaries,
            shuffle=True)
    else:
        train_sampler = promovits.data.sampler.RandomBucketSampler(
            train_dataset,
            promovits.BATCH_SIZE,
            boundaries)

    # Create loaders
    train_loader = torch.data.utils.DataLoader(
        train_dataset,
        num_workers=promovits.NUM_WORKERS,
        shuffle=False,
        pin_memory=gpu is not None,
        collate_fn=collate_fn,
        batch_sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        num_workers=promovits.NUM_WORKERS,
        shuffle=False,
        batch_size=promovits.BATCH_SIZE,
        pin_memory=gpu is not None,
        drop_last=False,
        collate_fn=collate_fn)

    return train_loader, valid_loader
