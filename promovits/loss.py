import torch


###############################################################################
# Loss functions
###############################################################################


def feature_matching(real_feature_maps, fake_feature_maps):
    """Feature matching loss"""
    loss = 0.
    iterator = zip(real_feature_maps, fake_feature_maps)
    for real_feature_map, fake_feature_map in iterator:
        for real, fake in zip(real_feature_map, fake_feature_map):
            loss += torch.mean(torch.abs(real.float().detach() - fake.float()))
    return 2. * loss


def discriminator(real_outputs, fake_outputs):
    """Discriminator loss"""
    real_losses = []
    fake_losses = []
    for real_output, fake_output in zip(real_outputs, fake_outputs):
        real_losses.append(torch.mean((1. - real_output.float()) ** 2.))
        fake_losses.append(torch.mean(fake_output.float() ** 2.))
    return sum(real_losses) + sum(fake_losses), real_losses, fake_losses


def generator(discriminator_outputs):
    """Generator adversarial loss"""
    losses = [
        torch.mean((1. - output.float()) ** 2.)
        for output in discriminator_outputs]
    return sum(losses), losses


def kl(z_p, logs_q, m_p, logs_p, z_mask):
    """KL-divergence loss"""
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2. * logs_p)
    return torch.sum(kl * z_mask) / torch.sum(z_mask)
