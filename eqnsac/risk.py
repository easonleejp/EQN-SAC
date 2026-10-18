"""EQN 不确定性诊断与本项目的 Actor 风险目标。

输入 quantiles 的维度为 [成员 K, 批量 B, 分位 N]，taus 为 [B, N]。
原 EQN 的两个方差是 Var_tau(E_k Z) 与 Var_k(E_tau Z)；这里与作者
官方实现一样报告标准差。自适应风险水平和尾部分歧惩罚是本项目扩展。
"""

from typing import NamedTuple

import torch


class RiskStatistics(NamedTuple):
    """每个字段均为 [B]；policy_value 用于训练，其余字段用于解释目标。"""

    policy_value: torch.Tensor
    aleatoric_std: torch.Tensor
    epistemic_std: torch.Tensor
    tail_disagreement: torch.Tensor
    cvar_level: torch.Tensor


def midpoint_quantiles(batch_size, count, device):
    """所有消融组共享 (i+0.5)/N 网格，避免比较时混入采样方式差异。"""
    return ((torch.arange(count, device=device) + 0.5) / count).expand(batch_size, -1)


def risk_statistics(quantiles, taus, cfg):
    """计算原 EQN 诊断和现有四组策略目标，不改变历史风险公式。

    Original: E_tau E_k Z。
    Adaptive-only: level=clip(exp(-beta * aleatoric_std), min_level, 1)，
        再对 tau <= level 的网格点取均值；level 明确停止梯度。
    Tail-only: 全分位均值减 lambda * tail_disagreement。
    Full: 自适应低尾均值减 lambda * tail_disagreement。

    tail_disagreement 是不同成员在相同低尾分位上的成对绝对差，
    不是原 EQN 的 epistemic_std。未强制分位单调时也不等同于精确 W1。
    低尾均值是硬网格近似；例如 N=32、level=.1 时使用3个中点。
    """
    mean_quantiles = quantiles.mean(0)              # [B, N] = E_k Z
    aleatoric_std = mean_quantiles.std(-1, unbiased=False)
    member_means = quantiles.mean(-1)              # [K, B] = E_tau Z
    epistemic_std = member_means.std(0, unbiased=False)

    use_adaptive = cfg.arm in ("adaptive_only", "full")
    use_tail_penalty = cfg.arm in ("tail_only", "full")
    cvar_level = (
        torch.exp(-cfg.adaptive_beta * aleatoric_std.detach()).clamp(
            cfg.adaptive_min_level, 1
        )
        if use_adaptive
        else torch.ones_like(aleatoric_std)
    )
    selected_quantiles = taus <= cvar_level[:, None]
    selected_quantiles[:, 0] = True  # Even a very low level retains one quantile.
    policy_value = (mean_quantiles * selected_quantiles).sum(-1) / selected_quantiles.sum(-1)

    tail_mask = taus <= cfg.tail_fraction
    tail_mask[:, 0] = True
    first, second = torch.triu_indices(
        len(quantiles), len(quantiles), offset=1, device=quantiles.device
    )
    pair_disagreement = (
        (quantiles[first] - quantiles[second]).abs() * tail_mask[None]
    ).sum(-1) / tail_mask.sum(-1)[None]
    tail_disagreement = pair_disagreement.mean(0)
    if use_tail_penalty:
        policy_value = policy_value - cfg.tail_lambda * tail_disagreement

    return RiskStatistics(
        policy_value, aleatoric_std, epistemic_std, tail_disagreement, cvar_level
    )


def risk_value(z, tau, cfg):
    """兼容旧四元组接口；第三项一直是尾部分歧，不是原 EQN 认知标准差。

    新代码请使用 risk_statistics 的具名字段。
    """
    result = risk_statistics(z, tau, cfg)
    return (
        result.policy_value, result.aleatoric_std,
        result.tail_disagreement, result.cvar_level,
    )
