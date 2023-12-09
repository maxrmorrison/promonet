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

    def __init__(
        self,
        dataset,
        max_frames=promonet.MAX_TRAINING_FRAMES,
        variable_batch=promonet.VARIABLE_BATCH
    ):
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

        # Iterate over length-partitioned buckets
        batches = []
        for bucket in self.buckets:

            # Shuffle bucket
            bucket = bucket[
                torch.randperm(len(bucket), generator=generator).tolist()]

            # Variable batch size
            if self.variable_batch:
                batch = []
                max_length = 0
                for index, length in bucket:
                    max_length = max(max_length, length)
                    if (
                        batch and
                        (len(batch) + 1) * max_length > self.max_frames
                    ):
                        batches.append(batch)
                        max_length = 0
                        batch = []
                    else:
                        batch.append(index)

            # Constant batch size
            else:
                batches.extend([
                    bucket[i:i + promonet.BATCH_SIZE, 0]
                    for i in range(0, len(bucket), promonet.BATCH_SIZE)])

        # Shuffle
        return [
            batches[i] for i in
            torch.randperm(len(batches), generator=generator).tolist()]

    def set_epoch(self, epoch):
        self.epoch = epoch
