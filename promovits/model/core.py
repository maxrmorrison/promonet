import torch


###############################################################################
# Shared model utilities
###############################################################################


def convert_pad_shape(pad_shape):
    return [item for sublist in pad_shape[::-1] for item in sublist]


# TODO - is this a multiprocessing issue?
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


# def generate_path(duration, mask):
#     """
#     duration: [b, 1, t_x]
#     mask: [b, 1, t_y, t_x]
#     """
#     b, _, t_y, t_x = mask.shape
#     cum_duration = torch.cumsum(duration, -1)
#     cum_duration_flat = cum_duration.view(b * t_x)
#     path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
#     path = path.view(b, t_x, t_y)
#     pad_shape = convert_pad_shape([[0, 0], [1, 0], [0, 0]])
#     path = path - torch.nn.functional.pad(path, pad_shape)[:, :-1]
#     return path.unsqueeze(1).transpose(2, 3) * mask


def get_padding(kernel_size, dilation=1):
    """Compute the padding needed to perform same-size convolution"""
    return int((kernel_size * dilation - dilation) / 2)


# TODO - is this non-default initialization useful?
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def random_slice_segments(segments, lengths, segment_size):
    """Randomly slice segments along last dimension"""
    max_start_indices = lengths - segment_size + 1
    start_indices = torch.rand((len(segments),), device=segments.device)
    start_indices = (start_indices * max_start_indices).to(dtype=torch.long)
    segments = slice_segments(segments, start_indices, segment_size)
    return segments, start_indices


def sequence_mask(length, max_length=None):
    """Compute a binary mask from sequence lengths"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def slice_segments(segments, start_indices, segment_size):
    """Slice segments along last dimension"""
    slices = torch.zeros_like(segments[..., :segment_size])
    iterator = enumerate(zip(segments, start_indices))
    for i, (segment, start_index) in iterator:
        slices[i] = segment[..., start_index:start_index + segment_size]
    return slices


def weight_norm_conv1d(*args, **kwargs):
    """Construct Conv1d layer with weight normalization"""
    return torch.nn.utils.weight_norm(torch.nn.Conv1d(*args, **kwargs))


def weight_norm_conv2d(*args, **kwargs):
    """Construct Conv2d layer with weight normalization"""
    return torch.nn.utils.weight_norm(torch.nn.Conv2d(*args, **kwargs))
