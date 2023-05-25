import math

import torch

import promonet


###############################################################################
# Duration prediction
###############################################################################


class DurationPredictor(torch.nn.Module):

    def __init__(self, in_channels, filter_channels, p_dropout=.5, n_flows=4):
        super().__init__()
        self.log_flow = promonet.model.modules.Log()
        self.flows = torch.nn.ModuleList()
        self.flows.append(promonet.model.modules.ElementwiseAffine(2))
        for _ in range(n_flows):
            self.flows.append(promonet.model.modules.ConvFlow(
                2,
                filter_channels,
                promonet.KERNEL_SIZE,
                n_layers=3))
            self.flows.append(promonet.model.modules.Flip())

        self.post_pre = torch.nn.Conv1d(1, filter_channels, 1)
        self.post_proj = torch.nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = promonet.model.modules.DDSConv(
            filter_channels,
            promonet.KERNEL_SIZE,
            n_layers=3,
            p_dropout=p_dropout)
        self.post_flows = torch.nn.ModuleList()
        self.post_flows.append(promonet.model.modules.ElementwiseAffine(2))
        for _ in range(4):
            self.post_flows.append(promonet.model.modules.ConvFlow(
                2,
                filter_channels,
                promonet.KERNEL_SIZE,
                n_layers=3))
            self.post_flows.append(promonet.model.modules.Flip())

        self.pre = torch.nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = torch.nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = promonet.model.modules.DDSConv(
            filter_channels,
            promonet.KERNEL_SIZE,
            n_layers=3,
            p_dropout=p_dropout)
        if promonet.GLOBAL_CHANNELS != 0:
            self.cond = torch.nn.Conv1d(
                promonet.GLOBAL_CHANNELS,
                filter_channels,
                1)

    def forward(
            self,
            x,
            feature_mask,
            w=None,
            g=None,
            reverse=False,
            noise_scale=1.):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, feature_mask)
        x = self.proj(x) * feature_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, feature_mask)
            h_w = self.post_proj(h_w) * feature_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(
                device=x.device, dtype=x.dtype) * feature_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, feature_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * feature_mask
            z0 = (w - u) * feature_mask
            logdet_tot_q += torch.sum(
                (torch.nn.functional.logsigmoid(z_u) +
                 torch.nn.functional.logsigmoid(-z_u)) * feature_mask,
                [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) +
                             (e_q ** 2)) * feature_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, feature_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, feature_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2))
                            * feature_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(
                device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, feature_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


###############################################################################
# Duration prediction
###############################################################################


class ConvFlow(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            filter_channels,
            kernel_size,
            n_layers,
            num_bins=10,
            tail_bound=5.0):
        super().__init__()
        self.filter_channels = filter_channels
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = torch.nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = promonet.model.modules.DDSConv(
            filter_channels,
            kernel_size,
            n_layers,
            p_dropout=0.)
        self.proj = torch.nn.Conv1d(
            filter_channels,
            self.half_channels * (num_bins * 3 - 1),
            1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels]*2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)

        unnormalized_widths = h[..., :self.num_bins] / \
            math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins:2 *
                                 self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = h[..., 2 * self.num_bins:]

        x1, logabsdet = promonet.model.spline.piecewise_rational_quadratic(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails='linear',
            tail_bound=self.tail_bound)

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x
