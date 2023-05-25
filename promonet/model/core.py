import torch


###############################################################################
# Shared model components
###############################################################################


class LayerNorm(torch.nn.Module):

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return torch.nn.functional.layer_norm(
            x.transpose(1, -1),
            (self.channels,),
            self.gamma,
            self.beta,
            self.eps).transpose(1, -1)


class Flip(torch.nn.Module):

    def forward(self, x, mask, g=None, reverse=False):
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return x


###############################################################################
# Shared model utilities
###############################################################################


def get_padding(kernel_size, dilation=1, stride=1):
    """Compute the padding needed to perform same-size convolution"""
    return int((kernel_size * dilation - dilation - stride + 1) / 2)


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
