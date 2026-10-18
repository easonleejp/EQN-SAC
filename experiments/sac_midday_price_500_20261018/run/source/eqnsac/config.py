"""One explicit schema; reject misspelled or historical configuration switches."""

from dataclasses import dataclass, fields
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    """单次实验配置；默认值保留已有实验口径，不代表已调优的参数。

    initial_alpha/target_entropy属于SAC熵温度；adaptive_min_level属于
    CVaR分位水平。两者物理含义不同，不混用alpha命名解释。
    """

    # Experiment identity and budget.
    seed: int = 42
    device: str = "auto"
    arm: str = "original"
    episodes: int = 5000
    # Network capacity and fixed randomized-prior ensemble.
    hidden_dim: int = 256
    ensemble_size: int = 5
    quantiles: int = 32
    embedding_dim: int = 64
    prior_scale: float = 300.0
    # Replay sampling; membership is fixed when each transition is inserted.
    bootstrap_probability: float = 0.5
    batch_size: int = 128
    replay_size: int = 200000
    warmup_steps: int = 200
    # Soft Bellman target. gamma=1 is a finite 24-hour terminal task here.
    gamma: float = 1.0
    target_tau: float = 0.005
    reward_scale: float = 0.001
    # Optimizers and per-role entropy temperatures.
    price_lr: float = 0.0003
    storage_lr: float = 0.001
    critic_lr: float = 0.0003
    alpha_lr: float = 0.00003
    initial_alpha: float = 0.2
    storage_alpha_max: float = 0.2
    target_entropy: float = -1.0
    # Observation/reward choices; keep these fixed across method ablations.
    storage_observation: str = "compact10"
    storage_reward_mix: float = 0.1
    # Project extensions beyond original EQN; beta multiplies a std, not a variance.
    adaptive_beta: float = 0.01
    adaptive_min_level: float = 0.1
    tail_fraction: float = 0.1
    tail_lambda: float = 0.1
    # Scenario distribution and daily single-source scale bounds.
    level_probabilities: tuple = (0.2, 0.6, 0.2)
    response_rng: str = "paired_fixed"
    scenario_mode: str = "three_source"
    load_scale_bound: float = 0.05
    wind_scale_bound: float = 0.05
    price_scale_bound: float = 0.03
    reference_seed: int = 20260914
    # Output cadence, measured in complete episodes.
    evaluation_every: int = 500
    evaluation_cases: int = 27
    checkpoint_every: int = 500
    log_every: int = 50

    def validate(self):
        # Dataclass annotations do not enforce types, including for JSON inputs.
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is int:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(field.name + " must be an integer")
            elif field.type is float:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(field.name + " must be a finite number")
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError(field.name + " must be a finite number")
            elif field.type is str and not isinstance(value, str):
                raise ValueError(field.name + " must be a string")
        # Actor and quantile RNGs add offsets before calling torch.manual_seed.
        if not 0 <= self.seed <= 2**64 - 1 - 20001:
            raise ValueError("seed is outside the supported RNG range")
        if self.reference_seed < 0:
            raise ValueError("reference_seed must be nonnegative")
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
        probabilities = self.level_probabilities
        if not isinstance(probabilities, (list, tuple)) or len(probabilities) != 3:
            raise ValueError("Invalid level probabilities")
        if any(
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or not 0 <= p <= 1
            for p in probabilities
        ) or abs(sum(probabilities) - 1) > 1e-8:
            raise ValueError("Invalid level probabilities")
        return self


def load_config(path):
    values = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    return Config(**values).validate()
