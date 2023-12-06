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

    def __init__(self, dataset, max_frames = promonet.MAX_TRAINING_FRAMES, variable_batch = promonet.VARIABLE_BATCH):
        self.max_frames = max_frames
        self.epoch = 0
        self.length = len(dataset)
        self.buckets = dataset.buckets()
        self.variable_batch = variable_batch

    def __iter__(self):
        return iter(self.batch())

    def __len__(self):
        return len(self.batch())

    def batch(self):
        """Produces batch indices for one epoch"""
        # Deterministic shuffling based on epoch
        generator = torch.Generator()
        generator.manual_seed(promonet.RANDOM_SEED + self.epoch)

        # Make batches with roughly equal number of frames
        batches = []
        for max_length, bucket in self.buckets:

            # Shuffle bucket
            bucket = bucket[
                torch.randperm(len(bucket), generator=generator).tolist()]

            # Get current batch size
            if self.variable_batch:
                size = self.max_frames // max_length
                #print(size)
            else:
                size = promonet.BATCH_SIZE

            # Make batches
            batches.extend(
                [bucket[i:i + size] for i in range(0, len(bucket), size)])

        # Shuffle
        return [
            batches[i] for i in
            torch.randperm(len(batches), generator=generator).tolist()]

    def set_epoch(self, epoch):
        self.epoch = epoch
