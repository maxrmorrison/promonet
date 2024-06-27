import torch

import promonet


###############################################################################
# Adversarial loss functions
###############################################################################


def feature_matching(real_feature_maps, fake_feature_maps):
    """Feature matching loss"""
    loss = 0.
    iterator = zip(real_feature_maps, fake_feature_maps)
    for real_feature_map, fake_feature_map in iterator:

        # Maybe omit first activation layers from feature matching loss
        if promonet.FEATURE_MATCHING_OMIT_FIRST:
            real_feature_map = real_feature_map[1:]
            fake_feature_map = fake_feature_map[1:]

        # Aggregate
        for real, fake in zip(real_feature_map, fake_feature_map):
            loss += torch.mean(torch.abs(real.float().detach() - fake.float()))

    return loss


def discriminator(real_outputs, fake_outputs):
    """Discriminator loss"""
    real_losses = []
    fake_losses = []
    for real_output, fake_output in zip(real_outputs, fake_outputs):
        if promonet.ADVERSARIAL_HINGE_LOSS:
            real_losses.append(torch.mean(torch.clamp(1. - real_output, min=0.)))
            fake_losses.append(torch.mean(torch.clamp(1 + fake_output, min=0.)))
        else:
            real_losses.append(torch.mean((1. - real_output) ** 2.))
            fake_losses.append(torch.mean(fake_output ** 2.))
    return sum(real_losses) + sum(fake_losses), real_losses, fake_losses


def generator(discriminator_outputs):
    """Generator adversarial loss"""
    if promonet.ADVERSARIAL_HINGE_LOSS:
        losses = [
            torch.mean(torch.clamp(1. - output, min=0.))
            for output in discriminator_outputs]
    else:
        losses = [
            torch.mean((1. - output) ** 2.)
            for output in discriminator_outputs]
    return sum(losses), losses


###############################################################################
# Spectral loss functions
###############################################################################


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    magnitude = torch.abs(
        torch.stft(
            x,
            fft_size,
            hop_size,
            win_length,
            window,
            return_complex=True))
    return torch.sqrt(torch.clamp(magnitude, min=1e-7))


class SpectralConvergence(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self,
        device,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window='hann_window'
    ):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length).to(device)

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, 1, T).
            y (Tensor): Groundtruth signal (B, 1, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(
            x.squeeze(1),
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window)
        y_mag = stft(
            y.squeeze(1),
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window)
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)


class MultiResolutionSpectralConvergence(torch.nn.Module):

    def __init__(
        self,
        device,
        fft_sizes=[2560, 1280, 640, 320, 160, 80],
        hop_sizes=[640, 320, 160, 80, 40, 20],
        win_lengths=[2560, 1280, 640, 320, 160, 80],
        window='hann_window'
    ):
        super().__init__()
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [SpectralConvergence(device, fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, 1, T).
            y (Tensor): Groundtruth signal (B, 1, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value
        """
        sc_loss = 0.0
        for stft_loss in self.stft_losses:
            sc_loss += stft_loss(x, y)
        return  sc_loss / len(self.stft_losses)


###############################################################################
# Time-domain loss functions
###############################################################################


def signal(y_true, y_pred):
    """Waveform loss function"""
    t = y_true / (1e-15 + torch.norm(y_true, dim=-1, p=2, keepdim=True))
    p = y_pred / (1e-15 + torch.norm(y_pred, dim=-1, p=2, keepdim=True))
    return torch.mean(1. - torch.sum(p * t, dim=-1))
