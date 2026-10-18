"""Auditable parameter overrides; no changes to response or profit equations.

Retail bounds, original TOU, hourly response reference, dispatch constraints and
settlement rules remain the source values. Each run saves its effective inputs.
"""

import json
import math

from eqnsac.config import ROOT
from eqnsac.environment import Environment
from eqnsac.scenarios import GeneratorConfig


RESPONSE_KEYS = {"curvature_mean", "curvature_std", "demand_noise_std", "min_curvature"}
STORAGE_KEYS = {"soc_min", "soc_max", "soc_init", "p_ch_max", "p_dis_max", "eta_ch", "eta_dis", "c_es"}


def validate_overrides(cfg):
    for values, allowed, label in ((cfg.response_parameters, RESPONSE_KEYS, "response"),
                                   (cfg.storage_parameters, STORAGE_KEYS, "storage")):
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"Unsupported {label} parameter override")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in values.values()):
            raise ValueError(f"{label} parameters must be finite numbers")
    if not 0 <= cfg.renewable_penetration <= 1:
        raise ValueError("Renewable penetration must be in [0, 1]")
    demand = cfg.response_parameters
    for key in ("curvature_mean", "min_curvature"):
        if key in demand and demand[key] <= 0:
            raise ValueError(key + " must be positive")
    for key in ("curvature_std", "demand_noise_std"):
        if key in demand and demand[key] < 0:
            raise ValueError(key + " must be nonnegative")


def make_environment(cfg):
    validate_overrides(cfg)
    spec = json.loads((ROOT / "data/environment.json").read_text("utf-8"))
    spec["demand"].update(cfg.response_parameters)
    spec["storage"].update(cfg.storage_parameters)
    return Environment(spec=spec, generator_config=GeneratorConfig(target_penetration=cfg.renewable_penetration))
