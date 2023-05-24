import torch

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
        real_losses.append(torch.mean((1. - real_output) ** 2.))
        fake_losses.append(torch.mean(fake_output ** 2.))
    return sum(real_losses) + sum(fake_losses), real_losses, fake_losses


def generator(discriminator_outputs):
    """Generator adversarial loss"""
    losses = [
        torch.mean((1. - output) ** 2.)
        for output in discriminator_outputs]
    return sum(losses), losses


def kl(prior, true_logstd, predicted_mean, predicted_logstd, latent_mask):
    """KL-divergence loss"""
    divergence = predicted_logstd - true_logstd - 0.5 + \
        0.5 * ((prior - predicted_mean) ** 2) * \
        torch.exp(-2. * predicted_logstd)
    return torch.sum(divergence * latent_mask) / torch.sum(latent_mask)
