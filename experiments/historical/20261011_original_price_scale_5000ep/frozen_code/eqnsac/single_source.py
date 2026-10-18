"""One daily multiplier on the selected source; all other inputs are fixed."""

import numpy as np


def sample_case(env, cfg, seed, levels=("medium", "medium", "medium")):
    if cfg.scenario_mode == "three_source":
        return env.generator.sample(seed, *levels)
    # The exported nominal input contains no wind curve. Freeze one reproducible
    # low-level wind realization rather than resampling wind each episode.
    reference = env.generator.sample(cfg.reference_seed, "low", "low", "low")
    if cfg.scenario_mode not in ("load_scale", "wind_scale", "price_scale"):
        raise ValueError("Unsupported single-source mode")
    bound = {"load_scale": cfg.load_scale_bound, "wind_scale": cfg.wind_scale_bound,
             "price_scale": cfg.price_scale_bound}[cfg.scenario_mode]
    delta = float(np.random.default_rng(seed).uniform(-bound, bound))
    load_delta = delta if cfg.scenario_mode == "load_scale" else 0.0
    wind_delta = delta if cfg.scenario_mode == "wind_scale" else 0.0
    price_delta = delta if cfg.scenario_mode == "price_scale" else 0.0
    load = env.generator.load * (1 + load_delta)
    if cfg.scenario_mode == "wind_scale":
        if reference["wind_power"].max() * (1 + bound) > env.generator.wind_capacity:
            raise ValueError("Wind scaling exceeds capacity; do not silently clip curve shape")
        reference["wind_power"] = reference["wind_power"] * (1 + wind_delta)
        # This is a power-profile experiment, not a wind-speed perturbation.
        reference["reference_wind_speed"] = reference.pop("wind_speed")
    reference.update(
        seed=int(seed), scenario_mode=cfg.scenario_mode, load_delta=load_delta,
        wind_delta=wind_delta,
        price_delta=price_delta,
        reference_seed=cfg.reference_seed,
        load_level="daily_uniform" if cfg.scenario_mode == "load_scale" else "nominal_fixed",
        wind_level="daily_uniform" if cfg.scenario_mode == "wind_scale" else "fixed",
        price_level="daily_uniform" if cfg.scenario_mode == "price_scale" else "nominal_fixed", load_shift_hours=0.0,
        spike_time=-1, spike_multiplier_raw=1.0,
        baseline_load=load,
        class_baseline_load=load[:, None] * env.generator.shares,
        grid_price=env.generator.price * (1 + price_delta),
    )
    return reference
