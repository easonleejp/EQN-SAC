"""One explicit schema; reject misspelled or historical configuration switches."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    arm: str = "original"
    episodes: int = 5000
    hidden_dim: int = 256
    ensemble_size: int = 5
    quantiles: int = 32
    embedding_dim: int = 64
    prior_scale: float = 300.0
    bootstrap_probability: float = 0.5
    batch_size: int = 128
    replay_size: int = 200000
    warmup_steps: int = 200
    gamma: float = 1.0
    target_tau: float = 0.005
    reward_scale: float = 0.001
    price_lr: float = 0.0003
    storage_lr: float = 0.001
    critic_lr: float = 0.0003
    alpha_lr: float = 0.00003
    initial_alpha: float = 0.2
    storage_alpha_max: float = 0.2
    target_entropy: float = -1.0
    storage_observation: str = "compact10"
    storage_reward_mix: float = 0.1
    adaptive_beta: float = 0.01
    adaptive_min_level: float = 0.1
    tail_fraction: float = 0.1
    tail_lambda: float = 0.1
    level_probabilities: tuple = (0.2, 0.6, 0.2)
    response_rng: str = "paired_fixed"
    scenario_mode: str = "three_source"
    load_scale_bound: float = 0.05
    wind_scale_bound: float = 0.05
    price_scale_bound: float = 0.03
    reference_seed: int = 20260914
    evaluation_every: int = 500
    evaluation_cases: int = 27
    checkpoint_every: int = 500
    log_every: int = 50

    def validate(self):
        if self.arm not in ("original", "adaptive_only", "tail_only", "full"):
            raise ValueError("Unknown arm")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Unknown device")
        if self.storage_observation not in ("compact10", "causal15"):
            raise ValueError("Unknown observation")
        if self.response_rng not in ("paired_fixed", "per_case"):
            raise ValueError("Unknown response RNG")
        if self.scenario_mode not in ("three_source", "load_scale", "wind_scale", "price_scale"):
            raise ValueError("Unknown scenario mode")
        if not 0 <= self.load_scale_bound < 1:
            raise ValueError("Invalid load scale bound")
        if not 0 <= self.wind_scale_bound < 1:
            raise ValueError("Invalid wind scale bound")
        if not 0 <= self.price_scale_bound < 1:
            raise ValueError("Invalid price scale bound")
        if self.scenario_mode != "three_source" and self.response_rng != "paired_fixed":
            raise ValueError("Single-source experiment requires fixed response RNG")
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(name)
        for name in (
            "episodes",
            "hidden_dim",
            "ensemble_size",
            "quantiles",
            "embedding_dim",
            "batch_size",
            "replay_size",
            "evaluation_every",
            "evaluation_cases",
            "checkpoint_every",
            "log_every",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")
        if self.ensemble_size < 2 or self.replay_size < self.batch_size:
            raise ValueError("Invalid ensemble/replay")
        if not 0 < self.gamma <= 1 or not 0 < self.target_tau <= 1:
            raise ValueError("Invalid discount/update")
        if (
            not 0 < self.bootstrap_probability <= 1
            or not 0 <= self.storage_reward_mix <= 1
        ):
            raise ValueError("Invalid probability/mix")
        if not 0 < self.adaptive_min_level <= 1 or not 0 < self.tail_fraction <= 1:
            raise ValueError("Invalid tail")
        if (
            self.prior_scale < 0
            or self.adaptive_beta < 0
            or self.tail_lambda < 0
            or self.warmup_steps < 0
        ):
            raise ValueError("Negative parameter")
        for name in (
            "price_lr",
            "storage_lr",
            "critic_lr",
            "alpha_lr",
            "initial_alpha",
            "storage_alpha_max",
            "reward_scale",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(name)
        if (
            len(self.level_probabilities) != 3
            or min(self.level_probabilities) < 0
            or abs(sum(self.level_probabilities) - 1) > 1e-8
        ):
            raise ValueError("Invalid level probabilities")
        return self


def load_config(path):
    values = json.loads(Path(path).read_text("utf-8"))
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    return Config(**values).validate()
