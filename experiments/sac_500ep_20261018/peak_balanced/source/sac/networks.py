"""Continuous-action SAC networks, independent of the EQN implementation."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GaussianActor(nn.Module):
    """Joint two-coordinate tanh Gaussian; log probability sums both coordinates."""

    def __init__(self, observation_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, 2)
        self.log_std = nn.Linear(hidden_dim, 2)

    def sample(self, observations, deterministic=False, generator=None):
        features = self.trunk(observations)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-20, 2)
        if deterministic:
            noise = torch.zeros_like(mean)
        else:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device,
                                generator=generator)
        latent = mean + log_std.exp() * noise
        action = torch.tanh(latent)
        normal_log_prob = -0.5 * noise.square() - log_std - 0.5 * math.log(2 * math.pi)
        log_jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        joint_log_prob = (normal_log_prob - log_jacobian).sum(-1, keepdim=True)
        return action, joint_log_prob


class ScalarQ(nn.Module):
    """One scalar value for the full observation and joint raw action."""

    def __init__(self, observation_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations, actions):
        return self.network(torch.cat((observations, actions), dim=-1))
