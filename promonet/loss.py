import librosa
import torch
import torchutil

import promonet


###############################################################################
# Loss functions
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


def kl(prior, predicted_mean, predicted_logstd, true_logstd, lengths):
    """KL-divergence loss"""
    divergence = predicted_logstd - true_logstd - 0.5 + \
        0.5 * ((prior - predicted_mean) ** 2) * \
        torch.exp(-2. * predicted_logstd)
    mask = torchutil.mask.from_lengths(
        lengths
    ).unsqueeze(1).to(divergence.dtype)
    return torch.sum(divergence * mask) / torch.sum(mask)


def multimel(ground_truth, generated):
    #Cache hann windows
    if (
        not hasattr(multimel, 'windows') or
        multimel.dtype != ground_truth.dtype or
        multimel.device != ground_truth.device
    ):
        multimel.windows = [torch.hann_window(
            win_size,
            dtype= ground_truth.dtype,
            device= ground_truth.device) for win_size in promonet.MULTI_MEL_LOSS_WINDOWS]
        multimel.dtype = ground_truth.dtype
        multimel.device = ground_truth.device

    #Cache mel bases
    if not hasattr(multimel, 'mel_bases'):
        all_bases = []
        for win_size in promonet.MULTI_MEL_LOSS_WINDOWS:
            basis = librosa.filters.mel(
                sr=promonet.SAMPLE_RATE,
                n_fft=win_size,
                n_mels=promonet.NUM_MELS)
            basis = torch.from_numpy(basis)
            basis = basis.to(ground_truth.dtype).to(ground_truth.device)
            all_bases.append(basis)
        multimel.mel_bases = all_bases

    loss = 0.0
    hopsizes = [window // 4 for window in promonet.MULTI_MEL_LOSS_WINDOWS]
    for hopsize, window, window_size, mel_basis in zip(hopsizes, multimel.windows, promonet.MULTI_MEL_LOSS_WINDOWS, multimel.mel_bases):

        # Pad audio
        size = (promonet.NUM_FFT - hopsize) // 2
        gt_pad = torch.nn.functional.pad(ground_truth, (size, size), mode='reflect')
        gen_pad = torch.nn.functional.pad(generated, (size, size), mode='reflect')

        # Compute stft
        gt_stft = torch.stft(
            gt_pad.squeeze(1),
            window_size,
            hop_length=hopsize,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True)

        gen_stft = torch.stft(
            gen_pad.squeeze(1),
            window_size,
            hop_length=hopsize,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True)

        # if promonet.COMPLEX_SPECTROGRAM:
        #     pass
        # else:

        gt_stft = torch.view_as_real(gt_stft)
        gen_stft = torch.view_as_real(gen_stft)

        #Compute magnitude
        gt_spec = torch.sqrt(gt_stft.pow(2).sum(-1) + 1e-6)
        gen_spec = torch.sqrt(gen_stft.pow(2).sum(-1) + 1e-6)

        gt_spec = torch.log(torch.clamp(torch.matmul(mel_basis, gt_spec), min=1e-5))
        gen_spec = torch.log(torch.clamp(torch.matmul(mel_basis, gen_spec), min=1e-5))

        loss += torch.nn.functional.l1_loss(gt_spec, gen_spec)

    return loss
