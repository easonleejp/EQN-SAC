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
    """Validate the allowlist and merged physical limits; return the effective spec."""
    for values, allowed, label in ((cfg.response_parameters, RESPONSE_KEYS, "response"),
                                   (cfg.storage_parameters, STORAGE_KEYS, "storage")):
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"Unsupported {label} parameter override")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in values.values()):
            raise ValueError(f"{label} parameters must be finite numbers")
    if (isinstance(cfg.renewable_penetration, bool)
            or not isinstance(cfg.renewable_penetration, (int, float))
            or not math.isfinite(cfg.renewable_penetration)
            or not 0 <= cfg.renewable_penetration <= 1):
        raise ValueError("Renewable penetration must be in [0, 1]")
    demand = cfg.response_parameters
    for key in ("curvature_mean", "min_curvature"):
        if key in demand and demand[key] <= 0:
            raise ValueError(key + " must be positive")
    for key in ("curvature_std", "demand_noise_std"):
        if key in demand and demand[key] < 0:
            raise ValueError(key + " must be nonnegative")

    # Cross-field limits must include defaults, because overrides may be partial.
    # SOC is stored energy in this model, so negative bounds are not meaningful.
    spec = json.loads((ROOT / "data/environment.json").read_text("utf-8"))
    spec["demand"].update(cfg.response_parameters)
    spec["storage"].update(cfg.storage_parameters)
    storage = spec["storage"]
    if not (0 <= storage["soc_min"] <= storage["soc_init"] <= storage["soc_max"]
            and storage["soc_min"] < storage["soc_max"]):
        raise ValueError("Storage requires 0 <= soc_min <= soc_init <= soc_max and positive usable capacity")
    if min(storage["p_ch_max"], storage["p_dis_max"]) <= 0:
        raise ValueError("Storage power limits must be positive")
    if not (0 < storage["eta_ch"] <= 1 and 0 < storage["eta_dis"] <= 1):
        raise ValueError("Storage efficiencies must be in (0, 1]")
    if storage["c_es"] < 0:
        raise ValueError("Storage throughput cost must be nonnegative")
    return spec


def make_environment(cfg):
    spec = validate_overrides(cfg)
    return Environment(spec=spec, generator_config=GeneratorConfig(target_penetration=cfg.renewable_penetration))
