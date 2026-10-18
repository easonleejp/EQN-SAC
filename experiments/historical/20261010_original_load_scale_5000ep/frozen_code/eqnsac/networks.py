"""Tanh policies, state-dependent feasible actions, and implicit quantile critics."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, indices, hidden, offset=None, scale=None):
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices))
        self.register_buffer(
            "offset", torch.tensor(offset or [0.0] * len(indices), dtype=torch.float32)
        )
        self.register_buffer(
            "scale", torch.tensor(scale or [1.0] * len(indices), dtype=torch.float32)
        )
        self.trunk = nn.Sequential(
            nn.Linear(len(indices), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.mean = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)

    def sample(self, state, deterministic=False, generator=None):
        h = self.trunk((state[:, self.indices] - self.offset) / self.scale)
        mu = self.mean(h)
        log_std = self.log_std(h).clamp(-20, 2)
        if deterministic:
            z = mu
        else:
            z = mu + log_std.exp() * torch.randn(
                mu.shape, device=mu.device, generator=generator
            )
        action = torch.tanh(z)
        logp = (
            -0.5 * ((z - mu) / log_std.exp()).square()
            - log_std
            - 0.5 * math.log(2 * math.pi)
        )
        logp -= 2 * (math.log(2) - z - F.softplus(-2 * z))
        return action, logp


def storage_transform(state, raw, storage):
    b = storage
    energy = b.soc_min + state[:, 5:6].clamp(0, 1) * (b.soc_max - b.soc_min)
    after = energy * b.eta_self
    lo = -((after - b.soc_min) * b.eta_dis).clamp(0, b.p_dis_max) / b.p_dis_max
    hi = ((b.soc_max - after) / b.eta_ch).clamp(0, b.p_ch_max) / b.p_ch_max
    width = (hi - lo) * 0.5
    return lo + (raw + 1) * width, torch.log(width.clamp_min(1e-8))


class QuantileCritic(nn.Module):
    def __init__(self, hidden, embedding, prior_scale):
        super().__init__()
        self.prior_scale = prior_scale
        self.register_buffer(
            "cosine_indices", torch.arange(embedding, dtype=torch.float32)
        )

        def branch():
            return nn.ModuleDict(
                {
                    "sa": nn.Sequential(
                        nn.Linear(21, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                    ),
                    "tau": nn.Linear(embedding, hidden),
                    "out": nn.Sequential(
                        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
                    ),
                }
            )

        self.learned = branch()
        self.prior = branch()
        self.prior.requires_grad_(False)

    def forward(self, state, action, tau):
        sa = torch.cat((state, action), -1)
        phi = torch.cos(math.pi * tau[:, :, None] * self.cosine_indices)

        def evaluate(branch):
            return branch["out"](
                branch["sa"](sa)[:, None, :] * F.relu(branch["tau"](phi))
            ).squeeze(-1)

        value = evaluate(self.learned)
        # Freeze weights, NOT action derivatives through the prior branch.
        if self.prior_scale:
            value = value + self.prior_scale * evaluate(self.prior)
        return value


def quantile_loss(prediction, target, tau):
    delta = target[:, None, :] - prediction[:, :, None]
    huber = torch.where(delta.abs() <= 1, 0.5 * delta.square(), delta.abs() - 0.5)
    # Preserve historical loss scaling: sum over predicted quantiles.
    return (
        ((tau[:, :, None] - (delta.detach() < 0).float()).abs() * huber).sum(1).mean()
    )


def risk_value(z, tau, cfg):
    """Same midpoint grid for ALL arms; discrete stop-gradient risk level.

    This deliberately does not claim a continuous gradient through alpha.
    Tail score is quantile disagreement, not a guaranteed Wasserstein metric.
    """
    mean = z.mean(0)
    alea = mean.std(-1, unbiased=False)
    adaptive = cfg.arm in ("adaptive_only", "full")
    tail = cfg.arm in ("tail_only", "full")
    level = (
        torch.exp(-cfg.adaptive_beta * alea.detach()).clamp(cfg.adaptive_min_level, 1)
        if adaptive
        else torch.ones_like(alea)
    )
    mask = tau <= level[:, None]
    mask[:, 0] = True
    value = (mean * mask).sum(-1) / mask.sum(-1)
    tail_mask = tau <= cfg.tail_fraction
    tail_mask[:, 0] = True
    pair_i, pair_j = torch.triu_indices(len(z), len(z), offset=1, device=z.device)
    disagreement = ((z[pair_i] - z[pair_j]).abs() * tail_mask[None]).sum(
        -1
    ) / tail_mask.sum(-1)[None]
    epi = disagreement.mean(0)
    if tail:
        value = value - cfg.tail_lambda * epi
    return value, alea, epi, level
