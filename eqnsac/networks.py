"""神经网络与可微动作映射；风险目标在 risk.py 中单独定义。

Actor 输入为全局状态，但只读取各自 indices 指定的观测。
Critic 输入 [B, STATE_DIM + ACTION_DIM]，输出 [B, N] 分位价值。
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .environment import ACTION_DIM, FEATURE_INDEX, STATE_DIM
from .risk import risk_value  # Public compatibility import for older callers.


class Actor(nn.Module):
    """一维 tanh-Gaussian 策略；网络参数名保留以兼容现有模型权重。"""

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
        """返回 [B,1] 原始动作及其含 tanh 雅可比的 log probability。"""
        h = self.trunk((state[:, self.indices] - self.offset) / self.scale)
        mu = self.mean(h)
        log_std = self.log_std(h).clamp(-20, 2)
        if deterministic:
            noise = torch.zeros_like(mu)
            z = mu
        else:
            noise = torch.randn(
                mu.shape, dtype=mu.dtype, device=mu.device, generator=generator
            )
            z = mu + log_std.exp() * noise
        action = torch.tanh(z)
        # Recovering noise as (z - mu) / std loses precision when std is small.
        # Keep the sampled noise to evaluate the same reparameterized density.
        logp = (
            -0.5 * noise.square()
            - log_std
            - 0.5 * math.log(2 * math.pi)
        )
        logp -= 2 * (math.log(2) - z - F.softplus(-2 * z))
        return action, logp


def storage_transform(state, raw, storage):
    """将 [-1,1] 的策略输出映射至当前SOC可行域，正号表示充电。

    返回执行动作及 log|da_exec/da_raw|，后者须从策略 logp 中扣除。
    环境目前限定 dt=1、eta_self=1，与 Environment.feasible_bounds 一致。
    """
    b = storage
    soc_index = FEATURE_INDEX["soc"]
    energy = b.soc_min + state[:, soc_index:soc_index + 1].clamp(0, 1) * (b.soc_max - b.soc_min)
    after = energy * b.eta_self
    lo = -((after - b.soc_min) * b.eta_dis).clamp(0, b.p_dis_max) / b.p_dis_max
    hi = ((b.soc_max - after) / b.eta_ch).clamp(0, b.p_ch_max) / b.p_ch_max
    width = (hi - lo) * 0.5
    return lo + (raw + 1) * width, torch.log(width.clamp_min(1e-8))


class QuantileCritic(nn.Module):
    """Z_k(s,a,tau)=learned_k + prior_scale * frozen_prior_k，EQN式(14)。

    prior参数冻结，动作导数仍保留，Actor可通过两个分支获得梯度。
    """

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
                        nn.Linear(STATE_DIM + ACTION_DIM, hidden),
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
        """state [B,S]、action [B,A]、tau [B,N] -> 分位值 [B,N]。"""
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
    """EQN式(16)的成对分位Huber损失，当前kappa固定为1。

    prediction [B,N]、target [B,N′]、tau [B,N]。
    对预测分位求和，对目标分位及batch取均值；不是对两类分位都取均值。
    原EQN交通实验表I使用kappa=10，此处是项目既有超参数选择。
    """
    delta = target[:, None, :] - prediction[:, :, None]
    absolute = delta.abs()
    quadratic = absolute.clamp(max=1)
    # Squaring an unselected large delta inside torch.where can still produce
    # NaN gradients. Only square the bounded quadratic part of the Huber loss.
    huber = 0.5 * quadratic.square() + (absolute - quadratic)
    # Preserve historical loss scaling: sum over predicted quantiles.
    return (
        ((tau[:, :, None] - (delta.detach() < 0).float()).abs() * huber).sum(1).mean()
    )

