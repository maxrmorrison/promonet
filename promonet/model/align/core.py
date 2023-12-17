import numpy as np
import torch

from promonet.model.align.mas import maximum_path_c


def maximum_path(observations, mask):
    """Cython optimized Monotonic Alignment Search (MAS)"""
    # Cache device and type
    device = observations.device
    dtype = observations.dtype

    # Initialize placeholder
    path = np.zeros(observations.shape, dtype=np.int32)

    # Setup log observation probabilities
    observations = observations.data.cpu().numpy().astype(float)

    # Get bounds
    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)

    # Decode
    maximum_path_c(path, observations, t_t_max, t_s_max)

    # Convert to torch
    return torch.from_numpy(path).to(device=device, dtype=dtype)
