import torch

import promonet


###############################################################################
# Sampler selection
###############################################################################


def sampler(dataset, partition):
    """Create batch sampler"""
    # Deterministic random sampler for training
    if partition.startswith('train'):
        return Sampler(dataset)

    # Sample validation and test data sequentially
    elif partition.startswith('test') or partition.startswith('valid'):
        return torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset),
            1,
            False)

    else:
        raise ValueError(f'Partition {partition} is not defined')


###############################################################################
# Samplers
###############################################################################


class Sampler:

    def __init__(self, dataset):
        self.epoch = 0
        self.length = len(dataset)

    def __iter__(self):
        return iter(self.batch())

    def __len__(self):
        return len(self.batch())

    def batch(self):
        """Produces batch indices for one epoch"""
        # Deterministic shuffling based on epoch
        generator = torch.Generator()
        generator.manual_seed(promonet.RANDOM_SEED + self.epoch)

        # Shuffle
        indices = torch.randperm(self.length, generator=generator).tolist()

        # Make batches
        return [
            indices[i:i + promonet.BATCH_SIZE]
            for i in range(0, self.length, promonet.BATCH_SIZE)]

    def set_epoch(self, epoch):
        self.epoch = epoch
