"""Auditable parameter overrides; no changes to response or profit equations.

Retail bounds, original TOU, dispatch and settlement equations stay unchanged.
Optional hourly utility coefficients set response curvature and its effective
reference price; each run saves the coefficients and nominal reference table.
"""

import json
import math

from eqnsac.config import ROOT
from eqnsac.environment import Environment
from eqnsac.scenarios import GeneratorConfig


RESPONSE_KEYS = {"curvature_mean", "curvature_std", "demand_noise_std", "min_curvature",
                 "participation_noise_std"}
STORAGE_KEYS = {"soc_min", "soc_max", "soc_init", "p_ch_max", "p_dis_max", "eta_ch", "eta_dis", "c_es"}


def _apply_hourly_utility(spec, coefficients):
    """Validate 24 coefficient pairs and record their equivalent parameters.

    The original global curvature mean remains available as a fallback. When
    explicit hourly coefficients exist, their curvature means take precedence.
    The nominal reference table is for audit; Environment uses the current
    float32 class-load sum to compute the runtime reference for that same U_h.
    """
    if not isinstance(coefficients, dict):
        raise ValueError("hourly_utility must be a dictionary")
    if not coefficients:
        return
    if set(coefficients) != {"a", "b"}:
        raise ValueError("hourly_utility requires exactly a and b")
    for name in ("a", "b"):
        values = coefficients[name]
        if not isinstance(values, (list, tuple)) or len(values) != 24:
            raise ValueError("hourly_utility." + name + " must contain 24 finite numbers")
        for value in values:
            try:
                valid = (not isinstance(value, bool)
                         and isinstance(value, (int, float)) and math.isfinite(value))
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("hourly_utility." + name + " must contain 24 finite numbers")
    demand = spec["demand"]
    if any(a >= 0 or -a < demand["min_curvature"] for a in coefficients["a"]):
        raise ValueError("hourly_utility requires a < 0 and -a >= min_curvature")
    if demand["curvature_std"] != 0:
        raise ValueError("hourly_utility requires curvature_std=0")

    profile = json.loads((ROOT / "data/profile.json").read_text("utf-8"))
    reference = [
        float(b) + 2 * float(a) * float(load)
        for a, b, load in zip(coefficients["a"], coefficients["b"], profile["baseline_load"])
    ]
    if not all(math.isfinite(value) for value in reference):
        raise ValueError("hourly_utility produces nonfinite nominal reference prices")
    # Copy the user-provided coefficients: later mutation must not alter a run.
    # b is a utility intercept, not a retail action, so no price-bound clipping.
    spec["hourly_utility"] = {name: list(coefficients[name]) for name in ("a", "b")}
    demand["hourly_curvature_mean"] = [-float(a) for a in coefficients["a"]]
    spec["hourly_demand_reference"] = reference


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
    for key in ("curvature_std", "demand_noise_std", "participation_noise_std"):
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
    _apply_hourly_utility(spec, cfg.hourly_utility)
    return spec


def make_environment(cfg):
    spec = validate_overrides(cfg)
    return Environment(spec=spec, generator_config=GeneratorConfig(target_penetration=cfg.renewable_penetration))
