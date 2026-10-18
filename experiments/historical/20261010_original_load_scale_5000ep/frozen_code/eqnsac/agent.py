"""Two decentralized actors and two objective-specific centralized ensembles."""

import copy
import numpy as np
import torch
from torch import nn
from .environment import STORAGE_INDICES
from .networks import (
    Actor,
    QuantileCritic,
    storage_transform,
    quantile_loss,
    risk_value,
)


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
                    (0, 1, 2, 3),
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
        price, lp = self.actors[0].sample(state, deterministic, self.action_rng)
        raw, ls = self.actors[1].sample(state, deterministic, self.action_rng)
        storage, jac = storage_transform(state, raw, self.storage)
        return torch.cat((price, storage), -1), [lp, ls - jac]

    @torch.no_grad()
    def act(self, state, deterministic=False):
        return (
            self.sample(
                torch.as_tensor(state, device=self.device)[None], deterministic
            )[0][0]
            .cpu()
            .numpy()
        )

    def update(self, replay):
        c = self.cfg
        losses = []
        alpha = self.log_alpha.exp().detach()
        for member in range(c.ensemble_size):
            batch = replay.sample(c.batch_size, self.device, member)
            s, a, r, ns, d = [
                batch[k] for k in ("state", "action", "reward", "next_state", "done")
            ]
            tau = torch.rand(
                (len(s), c.quantiles), device=self.device, generator=self.tau_rng
            )
            tau_next = torch.rand(
                (len(s), c.quantiles), device=self.device, generator=self.tau_rng
            )
            with torch.no_grad():
                na, logps = self.sample(ns)
            for objective in range(2):
                with torch.no_grad():
                    target = r[
                        :, objective : objective + 1
                    ] * c.reward_scale + c.gamma * (1 - d) * (
                        self.targets[objective][member](ns, na, tau_next)
                        - alpha[objective] * logps[objective]
                    )
                prediction = self.critics[objective][member](s, a, tau)
                loss = quantile_loss(prediction, target, tau)
                opt = self.critic_opts[objective][member]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.detach())
        s = replay.sample(c.batch_size, self.device)["state"]
        tau = (
            (torch.arange(c.quantiles, device=self.device) + 0.5) / c.quantiles
        ).expand(len(s), -1)
        actor_losses = []
        entropy_terms = []
        diagnostics = []
        for objective in range(2):
            ensemble = self.critics[objective]
            for q in ensemble:
                q.learned.requires_grad_(False)
            try:
                with torch.no_grad():
                    fixed, _ = self.sample(s)
                raw, logp = self.actors[objective].sample(s, generator=self.action_rng)
                if objective == 1:
                    selected, jac = storage_transform(s, raw, self.storage)
                    logp = logp - jac
                    joint = torch.cat((fixed[:, :1], selected), -1)
                else:
                    joint = torch.cat((raw, fixed[:, 1:]), -1)
                z = torch.stack([q(s, joint, tau) for q in ensemble])
                value, ua, ue, level = risk_value(z, tau, c)
                loss = (alpha[objective] * logp.squeeze(-1) - value).mean()
                opt = self.actor_opts[objective]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                actor_losses.append(loss.detach())
                entropy_terms.append(logp.detach().mean())
                diagnostics.extend(
                    [ua.detach().mean(), ue.detach().mean(), level.detach().mean()]
                )
            finally:
                for q in ensemble:
                    q.learned.requires_grad_(True)
        entropy = torch.stack(entropy_terms)
        alpha_loss = -(self.log_alpha * (entropy + c.target_entropy)).sum()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha[1].clamp_(max=float(np.log(c.storage_alpha_max)))
            for online, target in zip(
                self.critics.parameters(), self.targets.parameters()
            ):
                target.lerp_(online, c.target_tau)
        # Return device metrics; caller synchronizes only at episode boundaries.
        return torch.stack(
            [
                torch.stack(losses).mean(),
                *actor_losses,
                *diagnostics,
                *self.log_alpha.exp().detach(),
            ]
        )

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
