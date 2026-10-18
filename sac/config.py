"""Explicit configuration for the joint-action scalar-value SAC baseline."""

from dataclasses import dataclass, field, fields
from pathlib import Path
import json
import math


@dataclass
class Config:
    seed: int = 42
    device: str = "cpu"
    threads: int = 1
    episodes: int = 500
    hidden_dim: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    target_entropy: float = -2.0
    gamma: float = 1.0
    target_tau: float = 0.005
    batch_size: int = 128
    updates_per_step: int = 1
    replay_size: int = 50000
    warmup_steps: int = 1000
    reward_scale: float = 0.001
    observation: str = "full19"
    # Only response/storage/renewable parameters may change the business model.
    # The reward is always scaled original profit; it is not a configuration knob.
    response_parameters: dict = field(default_factory=dict)
    storage_parameters: dict = field(default_factory=dict)
    renewable_penetration: float = 0.2
    scenario_mode: str = "three_source"
    level_probabilities: tuple = (0.2, 0.6, 0.2)
    response_rng: str = "per_case"
    reference_seed: int = 20260914
    load_scale_bound: float = 0.05
    wind_scale_bound: float = 0.05
    price_scale_bound: float = 0.03
    evaluation_every: int = 50
    evaluation_cases: int = 27
    checkpoint_every: int = 100
    log_every: int = 50

    def validate(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is int and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f.name + " must be an integer")
            if f.type is float and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f.name + " must be finite")
            if f.type is str and not isinstance(value, str):
                raise ValueError(f.name + " must be a string")
        for name in ("threads", "episodes", "hidden_dim", "batch_size", "updates_per_step", "replay_size", "evaluation_cases", "checkpoint_every", "log_every"):
            if getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")
        for name in ("actor_lr", "critic_lr", "alpha_lr", "initial_alpha", "reward_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")
        if self.device not in ("cpu", "cuda") or self.observation not in ("full19", "causal11", "market14", "market18", "nominal11"):
            raise ValueError("Unsupported device or observation")
        if not 0 < self.gamma <= 1 or not 0 < self.target_tau <= 1:
            raise ValueError("Invalid gamma or target_tau")
        if not 0 <= self.seed <= 2**64 - 1 - 50001 or self.reference_seed < 0:
            raise ValueError("Seed is outside the supported RNG range")
        if self.warmup_steps < 0 or self.replay_size < self.batch_size or self.seed < 0:
            raise ValueError("Invalid exploration, replay or seed")
        if self.evaluation_every < 0:
            raise ValueError("evaluation_every must be nonnegative; zero disables internal evaluation")
        if self.scenario_mode not in ("three_source", "load_scale", "wind_scale", "price_scale", "deterministic_day"):
            raise ValueError("Unsupported scenario")
        if self.response_rng not in ("per_case", "paired_fixed"):
            raise ValueError("Unsupported response RNG")
        if (not isinstance(self.level_probabilities, (tuple, list)) or len(self.level_probabilities) != 3
                or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1 for p in self.level_probabilities)
                or abs(sum(self.level_probabilities) - 1) > 1e-8):
            raise ValueError("Invalid scenario probabilities")
        if any(not 0 <= getattr(self, name) < 1 for name in ("load_scale_bound", "wind_scale_bound", "price_scale_bound")):
            raise ValueError("Invalid daily scale")
        from .parameters import validate_overrides
        validate_overrides(self)
        if self.scenario_mode == "deterministic_day":
            # Require explicit zeroes: an omitted field must not silently retain
            # stochastic response defaults in an otherwise fixed-day experiment.
            for name in ("curvature_std", "demand_noise_std", "participation_noise_std"):
                if self.response_parameters.get(name) != 0:
                    raise ValueError("deterministic_day requires explicit " + name + "=0")
        return self


def load_config(path):
    values = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(values, dict) or set(values) - {f.name for f in fields(Config)}:
        raise ValueError("Unknown configuration fields or non-object JSON")
    return Config(**values).validate()
