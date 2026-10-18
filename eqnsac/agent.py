"""双Actor与按奖励目标划分的两套集中式EQN集成。

Price目标是总利润；Storage目标是环境返回的储能混合奖励。
每套有K个bootstrap成员，不是标准SAC的两支clipped-double-Q。
更新顺序固定为Critic -> 两个Actor -> 熵温度 -> target软更新。
"""

import copy
import numpy as np
import torch
from torch import nn
from .environment import ACTION_NAMES, PRICE_INDICES, STATE_DIM, STORAGE_INDICES
from .networks import (
    Actor,
    QuantileCritic,
    storage_transform,
    quantile_loss,
)
from .risk import midpoint_quantiles, risk_statistics

PRICE = 0
STORAGE = 1
UPDATE_METRIC_NAMES = (
    "critic_loss", "price_actor_loss", "storage_actor_loss",
    "price_aleatoric_std", "price_epistemic_std", "price_tail_disagreement", "price_risk_level",
    "storage_aleatoric_std", "storage_epistemic_std", "storage_tail_disagreement", "storage_risk_level",
    "price_alpha", "storage_alpha",
)


def checked_step(loss, optimizer, stage):
    """Stop at the failing update instead of propagating NaNs into a policy.

    This validates gradients without clipping or replacing them. A failed
    post-step check aborts training; the last saved checkpoint stays untouched.
    """
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    tensors = [loss.detach()] + [p.grad for p in parameters if p.grad is not None]
    if not torch.stack([torch.isfinite(t).all() for t in tensors]).all().item():
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(f"Nonfinite loss or gradient in {stage}")
    optimizer.step()
    # Adam's squared-gradient accumulator can overflow even with finite inputs.
    tensors = [p.detach() for p in parameters]
    tensors.extend(
        value
        for p in parameters
        for key, value in optimizer.state[p].items()
        if key != "step" and torch.is_tensor(value)
    )
    if not torch.stack([torch.isfinite(t).all() for t in tensors]).all().item():
        raise FloatingPointError(f"Nonfinite parameter or optimizer state in {stage}")


class Agent(nn.Module):
    def __init__(self, cfg, env):
        super().__init__()
        self.cfg = cfg
        self.storage = env.storage
        self.device = torch.device(
            "cuda"
            if cfg.device == "auto" and torch.cuda.is_available()
            else ("cpu" if cfg.device == "auto" else cfg.device)
        )
        torch.manual_seed(cfg.seed)
        self.actors = nn.ModuleList(
            [
                Actor(
                    PRICE_INDICES,
                    cfg.hidden_dim,
                    env.spec["price_actor_offset"],
                    env.spec["price_actor_scale"],
                ),
                Actor(STORAGE_INDICES[cfg.storage_observation], cfg.hidden_dim),
            ]
        )
        self.critics = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        QuantileCritic(
                            cfg.hidden_dim, cfg.embedding_dim, cfg.prior_scale
                        )
                        for _ in range(cfg.ensemble_size)
                    ]
                )
                for _ in range(2)
            ]
        )
        self.targets = copy.deepcopy(self.critics)
        self.targets.requires_grad_(False)
        self.log_alpha = nn.Parameter(
            torch.full((2,), float(np.log(cfg.initial_alpha)))
        )
        self.to(self.device)
        self.actor_opts = [
            torch.optim.Adam(a.parameters(), lr=lr)
            for a, lr in zip(self.actors, (cfg.price_lr, cfg.storage_lr))
        ]
        self.critic_opts = [
            [
                torch.optim.Adam(
                    [p for p in q.parameters() if p.requires_grad], lr=cfg.critic_lr
                )
                for q in ensemble
            ]
            for ensemble in self.critics
        ]
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        # Independent streams keep Actor draws comparable when risk code changes.
        self.action_rng = torch.Generator(device=self.device).manual_seed(
            cfg.seed + 10001
        )
        self.tau_rng = torch.Generator(device=self.device).manual_seed(cfg.seed + 20001)

    def sample(self, state, deterministic=False):
        """返回联合执行动作[B,2]和两Actor各自的logp[B,1]。"""
        price, lp = self.actors[PRICE].sample(state, deterministic, self.action_rng)
        raw, ls = self.actors[STORAGE].sample(state, deterministic, self.action_rng)
        storage, jac = storage_transform(state, raw, self.storage)
        return torch.cat((price, storage), -1), [lp, ls - jac]

    @torch.no_grad()
    def act(self, state, deterministic=False):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise FloatingPointError("Policy requires a finite 19-element state")
        action = (
            self.sample(
                torch.as_tensor(state, device=self.device)[None], deterministic
            )[0][0]
            .cpu()
            .numpy()
        )
        if not np.isfinite(action).all():
            raise FloatingPointError("Nonfinite policy action")
        return action

    def update(self, replay):
        """执行一次更新，返回与 UPDATE_METRIC_NAMES 同序的一维设备张量。

        保留bootstrap批次、随机流消费及优化器更新顺序。风险诊断中的
        epistemic_std与tail_disagreement分别记录，后者才参与Tail惩罚。
        """
        entropy_weights = self.log_alpha.exp().detach()
        critic_loss = self._update_critics(replay, entropy_weights)
        actor_losses, mean_log_probs, diagnostics = self._update_actors(
            replay, entropy_weights
        )
        self._update_temperature(mean_log_probs)
        self._update_targets()
        # Metrics stay on device; checked_step itself does synchronize for safety.
        return torch.stack([
            critic_loss, *actor_losses, *diagnostics,
            *self.log_alpha.exp().detach(),
        ])

    def _update_critics(self, replay, entropy_weights):
        """成员k从自己的固定bootstrap池采样，两奖励目标共享这批转移。"""
        cfg = self.cfg
        losses = []
        for member in range(cfg.ensemble_size):
            batch = replay.sample(cfg.batch_size, self.device, member)
            states = batch["state"]
            actions = batch["action"]
            rewards = batch["reward"]
            next_states = batch["next_state"]
            terminals = batch["done"]
            # Critic learns the FULL return distribution for every ablation arm.
            taus = torch.rand(
                (len(states), cfg.quantiles), device=self.device, generator=self.tau_rng
            )
            next_taus = torch.rand(
                (len(states), cfg.quantiles), device=self.device, generator=self.tau_rng
            )
            with torch.no_grad():
                next_actions, next_log_probs = self.sample(next_states)
            for objective in (PRICE, STORAGE):
                with torch.no_grad():
                    # Project's soft Bellman target uses this role's entropy only.
                    # Unlike original discrete EQN, next actions come from Actors.
                    target = rewards[:, objective:objective + 1] * cfg.reward_scale + (
                        cfg.gamma * (1 - terminals) * (
                            self.targets[objective][member](next_states, next_actions, next_taus)
                            - entropy_weights[objective] * next_log_probs[objective]
                        )
                    )
                predicted = self.critics[objective][member](states, actions, taus)
                loss = quantile_loss(predicted, target, taus)
                checked_step(
                    loss, self.critic_opts[objective][member],
                    f"critic objective={objective} ({ACTION_NAMES[objective]}) member={member}",
                )
                losses.append(loss.detach())
        return torch.stack(losses).mean()

    def _sample_actor_joint_action(self, states, objective):
        """固定另一Actor的动作，只对当前Actor的动作分量传播梯度。"""
        with torch.no_grad():
            fixed_actions, _ = self.sample(states)
        raw_action, log_prob = self.actors[objective].sample(
            states, generator=self.action_rng
        )
        if objective == STORAGE:
            executed_action, log_jacobian = storage_transform(states, raw_action, self.storage)
            joint_action = torch.cat((fixed_actions[:, :1], executed_action), -1)
            log_prob = log_prob - log_jacobian
        else:
            joint_action = torch.cat((raw_action, fixed_actions[:, 1:]), -1)
        return joint_action, log_prob

    def _update_actors(self, replay, entropy_weights):
        cfg = self.cfg
        states = replay.sample(cfg.batch_size, self.device)["state"]
        taus = midpoint_quantiles(len(states), cfg.quantiles, self.device)
        actor_losses, mean_log_probs, diagnostics = [], [], []
        for objective in (PRICE, STORAGE):
            ensemble = self.critics[objective]
            # Freeze Critic weights while retaining dQ/da, including prior dQ/da.
            for critic in ensemble:
                critic.learned.requires_grad_(False)
            try:
                joint_action, log_prob = self._sample_actor_joint_action(states, objective)
                quantiles = torch.stack([critic(states, joint_action, taus) for critic in ensemble])
                risk = risk_statistics(quantiles, taus, cfg)
                loss = (entropy_weights[objective] * log_prob.squeeze(-1) - risk.policy_value).mean()
                checked_step(loss, self.actor_opts[objective], f"actor objective={objective} ({ACTION_NAMES[objective]})")
                actor_losses.append(loss.detach())
                mean_log_probs.append(log_prob.detach().mean())
                diagnostics.extend([
                    risk.aleatoric_std.detach().mean(),
                    risk.epistemic_std.detach().mean(),
                    risk.tail_disagreement.detach().mean(),
                    risk.cvar_level.detach().mean(),
                ])
            finally:
                for critic in ensemble:
                    critic.learned.requires_grad_(True)
        return actor_losses, mean_log_probs, diagnostics

    def _update_temperature(self, mean_log_probs):
        """各Actor独立调节SAC温度；仅Storage保留既有的温度上限。"""
        log_probs = torch.stack(mean_log_probs)
        loss = -(self.log_alpha * (log_probs + self.cfg.target_entropy)).sum()
        checked_step(loss, self.alpha_opt, "entropy temperature")
        with torch.no_grad():
            self.log_alpha[STORAGE].clamp_(max=float(np.log(self.cfg.storage_alpha_max)))

    @torch.no_grad()
    def _update_targets(self):
        """target_tau是online权重所占比例，原网络键名与更新顺序保持不变。"""
        for online, target in zip(self.critics.parameters(), self.targets.parameters()):
            target.lerp_(online, self.cfg.target_tau)

    def optimizer_state(self):
        return {
            "actors": [x.state_dict() for x in self.actor_opts],
            "critics": [[x.state_dict() for x in row] for row in self.critic_opts],
            "alpha": self.alpha_opt.state_dict(),
            "action_rng": self.action_rng.get_state(),
            "tau_rng": self.tau_rng.get_state(),
        }

    def load_optimizer_state(self, state):
        for opt, value in zip(self.actor_opts, state["actors"]):
            opt.load_state_dict(value)
        for opts, values in zip(self.critic_opts, state["critics"]):
            for opt, value in zip(opts, values):
                opt.load_state_dict(value)
        self.alpha_opt.load_state_dict(state["alpha"])
        self.action_rng.set_state(state["action_rng"].cpu())
        self.tau_rng.set_state(state["tau_rng"].cpu())
