"""Standard scalar-reward SAC: one joint actor and clipped scalar twin critics.

The actor, replay and critics all use the same raw [-1, 1]^2 control coordinates.
Any state-dependent storage mapping belongs to the environment adapter; its
physical Jacobian is deliberately absent from this raw-coordinate entropy.
"""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .networks import GaussianActor, ScalarQ


def _require_finite(values, stage):
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError(f"Nonfinite {stage}")


def _checked_step(loss, optimizer, stage):
    _require_finite((loss,), f"loss in {stage}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    _require_finite([p.grad for p in parameters if p.grad is not None],
                    f"gradient in {stage}")
    optimizer.step()
    values = [p.detach() for p in parameters]
    values.extend(value for p in parameters for key, value in optimizer.state[p].items()
                  if key != "step" and torch.is_tensor(value))
    _require_finite(values, f"parameter or optimizer state in {stage}")


class Agent(nn.Module):
    def __init__(self, cfg, observation_dim):
        super().__init__()
        if not isinstance(observation_dim, int) or observation_dim <= 0:
            raise ValueError("observation_dim must be a positive integer")
        self.cfg = cfg
        self.observation_dim = observation_dim
        self.device = torch.device(
            "cuda" if cfg.device == "auto" and torch.cuda.is_available()
            else ("cpu" if cfg.device == "auto" else cfg.device)
        )
        # Initialization is reproducible without consuming a caller's CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            self.actor = GaussianActor(observation_dim, cfg.hidden_dim)
            self.critics = nn.ModuleList([
                ScalarQ(observation_dim, cfg.hidden_dim),
                ScalarQ(observation_dim, cfg.hidden_dim),
            ])
        self.targets = copy.deepcopy(self.critics)
        self.targets.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg.initial_alpha), dtype=torch.float32))
        self.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=cfg.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.action_rng = torch.Generator(device=self.device).manual_seed(cfg.seed + 10001)
        self.update_rng = torch.Generator(device=self.device).manual_seed(cfg.seed + 20001)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.observation_dim,) or not np.isfinite(obs).all():
            raise FloatingPointError("Policy requires a finite observation of the configured dimension")
        action, _ = self.actor.sample(torch.as_tensor(obs, device=self.device)[None],
                                      deterministic=deterministic, generator=self.action_rng)
        _require_finite((action,), "policy action")
        return action[0].cpu().numpy()

    @torch.no_grad()
    def _bellman_target(self, next_obs, rewards, terminals, alpha):
        next_actions, next_log_prob = self.actor.sample(next_obs, generator=self.update_rng)
        target_q1 = self.targets[0](next_obs, next_actions)
        target_q2 = self.targets[1](next_obs, next_actions)
        _require_finite((target_q1, target_q2), "SAC Bellman target critics")
        next_q = torch.minimum(target_q1, target_q2)
        target = rewards + self.cfg.gamma * (1 - terminals) * (next_q - alpha * next_log_prob)
        _require_finite((next_actions, next_log_prob, next_q, target), "SAC Bellman target")
        return target

    def _update_actor(self, observations, alpha):
        flags = [p.requires_grad for p in self.critics.parameters()]
        self.critics.requires_grad_(False)
        try:
            actions, log_prob = self.actor.sample(observations, generator=self.update_rng)
            q1 = self.critics[0](observations, actions)
            q2 = self.critics[1](observations, actions)
            _require_finite((actions, log_prob, q1, q2), "actor objective")
            q = torch.minimum(q1, q2)
            loss = (alpha * log_prob - q).mean()
            _checked_step(loss, self.actor_optimizer, "actor")
            return loss.detach(), log_prob.detach()
        finally:
            for parameter, flag in zip(self.critics.parameters(), flags):
                parameter.requires_grad_(flag)

    def update(self, replay):
        batch = replay.sample(self.cfg.batch_size, self.device)
        obs, actions, rewards, next_obs, terminals = (
            batch[name] for name in ("obs", "action", "reward", "next_obs", "done")
        )
        _require_finite(tuple(batch.values()), "replay batch")
        alpha = self.log_alpha.exp().detach()
        _require_finite((alpha,), "entropy temperature")
        target = self._bellman_target(next_obs, rewards, terminals, alpha)
        q1, q2 = self.critics[0](obs, actions), self.critics[1](obs, actions)
        q1_loss, q2_loss = F.mse_loss(q1, target), F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss
        _checked_step(critic_loss, self.critic_optimizer, "twin critics")
        actor_loss, log_prob = self._update_actor(obs, alpha)
        alpha_loss = -(self.log_alpha * (log_prob + self.cfg.target_entropy)).mean()
        _checked_step(alpha_loss, self.alpha_optimizer, "entropy temperature")
        with torch.no_grad():
            for target_parameter, parameter in zip(self.targets.parameters(), self.critics.parameters()):
                target_parameter.lerp_(parameter, self.cfg.target_tau)
        _require_finite([p.detach() for p in self.targets.parameters()], "target parameters")
        metrics = {
            "critic_loss": critic_loss.detach(), "q1_loss": q1_loss.detach(),
            "q2_loss": q2_loss.detach(), "actor_loss": actor_loss,
            "alpha_loss": alpha_loss.detach(), "alpha": self.log_alpha.exp().detach(),
            "entropy": -log_prob.mean(), "q1_mean": q1.detach().mean(),
            "q2_mean": q2.detach().mean(), "target_mean": target.mean(),
        }
        _require_finite(tuple(metrics.values()), "update metrics")
        return {name: float(value.cpu()) for name, value in metrics.items()}

    def optimizer_state(self):
        return {"actor": self.actor_optimizer.state_dict(),
                "critics": self.critic_optimizer.state_dict(),
                "alpha": self.alpha_optimizer.state_dict(),
                "action_rng": self.action_rng.get_state(),
                "update_rng": self.update_rng.get_state()}

    def load_optimizer_state(self, state):
        self.actor_optimizer.load_state_dict(state["actor"])
        self.critic_optimizer.load_state_dict(state["critics"])
        self.alpha_optimizer.load_state_dict(state["alpha"])
        self.action_rng.set_state(state["action_rng"].cpu())
        self.update_rng.set_state(state["update_rng"].cpu())
