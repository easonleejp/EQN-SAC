"""One daily load multiplier; all other exogenous inputs are fixed."""

import numpy as np


def sample_case(env, cfg, seed, levels=("medium", "medium", "medium")):
    if cfg.scenario_mode == "three_source":
        return env.generator.sample(seed, *levels)
    # The exported nominal input contains no wind curve. Freeze one reproducible
    # low-level wind realization rather than resampling wind each episode.
    reference = env.generator.sample(cfg.reference_seed, "low", "low", "low")
    delta = float(np.random.default_rng(seed).uniform(-cfg.load_scale_bound,
                                                     cfg.load_scale_bound))
    load = env.generator.load * (1 + delta)
    reference.update(
        seed=int(seed), scenario_mode="load_scale", load_delta=delta,
        reference_seed=cfg.reference_seed, load_level="daily_uniform",
        wind_level="fixed", price_level="nominal_fixed", load_shift_hours=0.0,
        spike_time=-1, spike_multiplier_raw=1.0,
        baseline_load=load,
        class_baseline_load=load[:, None] * env.generator.shares,
        grid_price=env.generator.price.copy(),
    )
    return reference
