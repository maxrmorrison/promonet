import torch

import promovits


###############################################################################
# Samplers
###############################################################################


class BucketSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        boundaries):
        super().__init__(dataset)
        self.boundaries = boundaries
        self.buckets, self.samples_per_bucket = self.create_buckets(
            dataset.lengths,
            boundaries)
        self.total_size = sum(self.samples_per_bucket)

    def __iter__(self):
        self.batches = make_batches(
            self.buckets,
            self.samples_per_bucket,
            self.epoch,
            False)
        return iter(self.batches)

    def __len__(self):
        """Retrieve the number of batches in an epoch"""
        return self.total_size // promovits.BATCH_SIZE

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(
        self,
        dataset,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle)
        self.boundaries = boundaries
        self.buckets, self.samples_per_bucket = create_buckets(
            dataset.lengths,
            boundaries,
            num_replicas)
        self.total_size = sum(self.samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
      self.batches = make_batches(
          self.buckets,
          self.samples_per_bucket,
          self.epoch,
          self.shuffle,
          self.rank,
          self.num_replicas)
      return iter(self.batches)

    def __len__(self):
        return self.num_samples // promovits.BATCH_SIZE


class RandomBucketSampler(torch.utils.data.RandomSampler):

    def __init__(self, dataset, boundaries):
        super().__init__(dataset)
        self.boundaries = boundaries
        self.buckets, self.samples_per_bucket = create_buckets(
            dataset.lengths,
            boundaries)
        self.total_size = sum(self.samples_per_bucket)

    def __iter__(self):
        self.batches = make_batches(
            self.buckets,
            self.samples_per_bucket,
            self.epoch)
        return iter(self.batches)

    def __len__(self):
        """Retrieve the number of batches in an epoch"""
        return self.total_size // promovits.BATCH_SIZE

    def set_epoch(self, epoch):
        self.epoch = epoch


###############################################################################
# Sampler utilities
###############################################################################


def bisect(x, boundaries, lo=0, hi=None):
    if hi is None:
        hi = len(boundaries) - 1

    if hi > lo:
        mid = (hi + lo) // 2
        if boundaries[mid] < x and x <= boundaries[mid+1]:
            return mid
        elif x <= boundaries[mid]:
            return bisect(x, boundaries, lo, mid)
        else:
            return bisect(x, boundaries, mid + 1, hi)
    return -1


def create_buckets(lengths, boundaries, num_replicas=1):
    buckets = [[] for _ in range(len(boundaries) - 1)]
    for i in range(len(lengths)):
        length = lengths[i]
        idx_bucket = bisect(length, boundaries)
        if idx_bucket != -1:
            buckets[idx_bucket].append(i)

    for i in range(len(buckets) - 1, 0, -1):
        if len(buckets[i]) == 0:
            buckets.pop(i)
            boundaries.pop(i+1)

    samples_per_bucket = []
    for i in range(len(buckets)):
        len_bucket = len(buckets[i])
        total_batch_size = num_replicas * promovits.BATCH_SIZE
        rem = (total_batch_size - (len_bucket %
                total_batch_size)) % total_batch_size
        samples_per_bucket.append(len_bucket + rem)
    return buckets, samples_per_bucket


def make_batches(
    buckets,
    samples_per_bucket,
    epoch,
    shuffle=True,
    rank=None,
    num_replicas=None):
    # Deterministic shuffling based on current epoch
    g = torch.Generator()
    g.manual_seed(epoch)

    indices = []
    if shuffle:
        for bucket in buckets:
            indices.append(torch.randperm(len(bucket), generator=g).tolist())
    else:
        for bucket in buckets:
            indices.append(list(range(len(bucket))))

    batches = []
    for i in range(len(buckets)):
        bucket = buckets[i]
        len_bucket = len(bucket)
        ids_bucket = indices[i]
        num_samples_bucket = samples_per_bucket[i]

        # Add extra samples to make it evenly divisible
        rem = num_samples_bucket - len_bucket
        ids_bucket = ids_bucket + ids_bucket * \
            (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

        # Subsample
        ids_bucket = ids_bucket[rank::num_replicas]

        # Batch
        for j in range(len(ids_bucket) // promovits.BATCH_SIZE):
            batch = [bucket[idx]
                     for idx in ids_bucket[j*promovits.BATCH_SIZE:(j+1)*promovits.BATCH_SIZE]]
            batches.append(batch)

    if shuffle:
        batch_ids = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in batch_ids]

    return batches
