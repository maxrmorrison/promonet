import torch


###############################################################################
# Shared model utilities
###############################################################################


def get_padding(kernel_size, dilation=1, stride=1):
    """Compute the padding needed to perform same-size convolution"""
    return int((kernel_size * dilation - dilation - stride + 1) / 2)


def random_slice_segments(segments, lengths, segment_size):
    """Randomly slice segments along last dimension"""
    max_start_indices = lengths - segment_size + 1
    start_indices = torch.rand((len(segments),), device=segments.device)
    start_indices = (start_indices * max_start_indices).to(dtype=torch.long)
    segments = slice_segments(segments, start_indices, segment_size)
    return segments, start_indices


def slice_segments(segments, start_indices, segment_size, fill_value=0.):
    """Slice segments along last dimension"""
    slices = torch.full_like(segments[..., :segment_size], fill_value)
    iterator = enumerate(zip(segments, start_indices))
    for i, (segment, start_index) in iterator:
        end_index = start_index + segment_size

        # Pad negative indices
        if start_index <= -segment_size:
            continue
        elif start_index < 0:
            start_index = 0

        # Slice
        slices[i, ..., -(end_index - start_index):] = \
            segment[..., start_index:end_index]

    return slices


def weight_norm_conv1d(*args, **kwargs):
    """Construct Conv1d layer with weight normalization"""
    return torch.nn.utils.weight_norm(torch.nn.Conv1d(*args, **kwargs))
