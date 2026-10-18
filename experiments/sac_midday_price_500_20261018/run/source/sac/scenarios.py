"""SAC scenario adapter; legacy stochastic modes retain their original sampler.

The deterministic day uses the original nominal load and wholesale price. The
profile contains no nominal wind trace, so wind is defined explicitly as a
constant power with daily energy equal to the configured share of nominal load.
It does not draw a random wind realization or imply an associated wind speed.
"""

import numpy as np

from eqnsac.single_source import sample_case as sample_stochastic_case


DETERMINISTIC_DAY_DEFINITION = (
    "One fixed 24-hour day: baseline load and wholesale price are copied from the "
    "original nominal profile; wind power is constant at mean nominal load times "
    "renewable_penetration. No scenario, shift, spike, or wind-speed draws are made. "
    "All response noise standard deviations are explicitly zero. reference_seed "
    "identifies this same day; repeats are not independent unseen scenarios."
)


def sample_case(env, cfg, seed, levels=("medium", "medium", "medium")):
    """Return fresh arrays; fixed-day inputs ignore the requested seed and levels."""
    if cfg.scenario_mode != "deterministic_day":
        return sample_stochastic_case(env, cfg, seed, levels)
    load = np.array(env.profile["baseline_load"], dtype=float, copy=True)
    price = np.array(env.profile["grid_price"], dtype=float, copy=True)
    wind = np.full(24, float(load.mean()) * cfg.renewable_penetration)
    return dict(
        seed=int(cfg.reference_seed),
        reference_seed=int(cfg.reference_seed),
        scenario_mode="deterministic_day",
        definition=DETERMINISTIC_DAY_DEFINITION,
        load_level="deterministic", wind_level="deterministic", price_level="deterministic",
        load_shift_hours=0.0, spike_time=-1, spike_multiplier_raw=1.0,
        load_delta=0.0, wind_delta=0.0, price_delta=0.0,
        baseline_load=load, grid_price=price, wind_power=wind,
        class_baseline_load=np.array(env.profile["class_baseline_load"], dtype=float, copy=True),
    )
