"""SAC control coordinates and observations for the unchanged profit objective.

Physics and user response are reused without changing the archived EQN task.
Raw storage control is affine-mapped into its feasible interval. SAC's replay,
critic and entropy all use the SAME raw [-1,1]^2 coordinates.
"""

import numpy as np
from .parameters import make_environment


class Task:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = make_environment(cfg)
        self.reference = make_environment(cfg)
        self.observation_dim = {"full19": 19, "causal11": 11, "market14": 14, "market18": 18}[cfg.observation]
        # Public nominal profiles, not the future realization of a sampled day.
        self.load_scale = float(self.env.norm["max_load"])
        self.price_center = float(np.mean(self.env.profile["grid_price"]))
        self.price_spread = float(np.std(self.env.profile["grid_price"]))

    def reset(self, case, response_seed):
        self.reference.reset(case, response_seed)
        self.reference_rows = []
        for hour in range(24):
            normalized = self.reference_price_action(hour)
            _, _, _, info = self.reference.step([normalized, 0.0])
            self.reference_rows.append(info)
        state = self.env.reset(case, response_seed)
        return self.observe(state)

    def reference_price_action(self, hour):
        low, high = self.env.user.price_min[0], self.env.user.price_max[0]
        price = self.env.profile["retail_price"][hour]
        if not low <= price <= high:
            raise ValueError("Reference retail tariff is outside allowed price bounds")
        return float(2 * (price - low) / (high - low) - 1)

    def observe(self, state):
        if self.cfg.observation in ("market14", "market18"):
            if self.env.t >= 24:
                return np.zeros(self.observation_dim, dtype=np.float32)
            hour = self.env.t
            env = self.env
            phase = 2 * np.pi * hour / 24
            # Known nominal TOU is information only; no action is tied to it.
            tou = 2 * (env.profile["retail_price"][hour] - env.user.price_min[0]) / (env.user.price_max[0] - env.user.price_min[0]) - 1
            wind_scale = max(float(env.generator.wind_capacity), 1.0)
            prices = [(env.price[max(0, hour - lag)] - self.price_center) / self.price_spread for lag in (0, 1, 2, 3)]
            winds = [env.wind[max(0, hour - lag)] / wind_scale for lag in (0, 1, 2, 3)]
            features = [prices[0], env.load[hour] / self.load_scale, winds[0], state[5],
                        (24 - hour) / 24, np.sin(phase), np.cos(phase), tou,
                        *prices[1:], *winds[1:]]
            if self.cfg.observation == "market18":
                # Public nominal quotes, not the future sampled price path.
                # At the day end, use the last known nominal hour (no next day).
                features += [(env.profile["grid_price"][min(hour + lag, 23)] - self.price_center) / self.price_spread
                             for lag in (1, 3, 6)]
                reference = env.spec["hourly_demand_reference"][hour]
                features += [2 * (reference - env.user.price_min[0]) / (env.user.price_max[0] - env.user.price_min[0]) - 1]
            return np.asarray(features, dtype=np.float32)
        # Previous prices/response do not affect transitions. Temperature is a
        # fixed function of time; forecasts are derived from current load/time.
        # Keep causal price/wind histories for the stochastic AR scenarios.
        obs = (state.copy() if self.cfg.observation == "full19"
               else state[[0, 1, 4, 5, 3, 10, 11, 12, 16, 17, 18]].copy())
        if self.cfg.observation == "causal11" and self.env.t < 24:
            # Gross demand matters for nonlinear price response; net demand and
            # wind contain this information, but exposing it avoids subtraction.
            obs[1] = self.env.load[self.env.t] / self.load_scale
            obs[4] = (24 - self.env.t) / 24
        return obs

    def step(self, raw_action):
        raw = np.asarray(raw_action, dtype=np.float32)
        if raw.shape != (2,) or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1 + 1e-6):
            raise ValueError("SAC requires a bounded finite two-dimensional raw action")
        raw = np.clip(raw, -1, 1)
        lo, hi = self.env.feasible_bounds()
        executed_storage = lo + (float(raw[1]) + 1) * 0.5 * (hi - lo)
        initial_energy = self.env.soc
        hour = self.env.t
        state, _, done, info = self.env.step([raw[0], executed_storage])
        grid = max(info["user_load"] + info["charge"] - info["discharge"] - float(self.env.wind[hour]), 0)
        net = info["user_load"] - float(self.env.wind[hour])
        baseline_profit = self.reference_rows[hour]["profit"]
        reward = info["profit"] * self.cfg.reward_scale
        info.update(
            raw_price_action=float(raw[0]), raw_storage_action=float(raw[1]),
            initial_soc=initial_energy, baseline_load=float(self.env.load[hour]),
            wind=float(self.env.wind[hour]), grid_price=float(self.env.price[hour]),
            grid_purchase=grid, net_load_without_storage=net,
            excess_energy=max(-(net + info["charge"] - info["discharge"]), 0),
            reference_profit=baseline_profit, learning_reward=float(reward),
        )
        if not np.isfinite(state).all() or not all(np.isfinite(v) for v in info.values()):
            raise FloatingPointError("Nonfinite task transition")
        return self.observe(state), float(reward), done, info


def daily_summary(rows, reference_rows, env):
    """Report physical and economic metrics separately from the learning reward."""
    def curve(values):
        a = np.asarray(values, dtype=float)
        return dict(peak=float(a.max()), valley=float(a.min()), peak_valley=float(np.ptp(a)),
                    mean=float(a.mean()), energy=float(a.sum()), std=float(a.std()),
                    par=float(a.max() / a.mean()) if a.mean() > 0 else None,
                    rms_ramp=float(np.sqrt(np.mean(np.diff(a)**2))))
    reference_grid = [max(r["user_load"] - rows[i]["wind"], 0) for i, r in enumerate(reference_rows)]
    curves = {
        "baseline_load": curve([r["baseline_load"] for r in rows]),
        "user_load": curve([r["user_load"] for r in rows]),
        "net_without_storage": curve([max(r["net_load_without_storage"], 0) for r in rows]),
        "grid": curve([r["grid_purchase"] for r in rows]),
        "reference_grid": curve(reference_grid),
    }
    return dict(profit=sum(r["profit"] for r in rows),
                reference_profit=sum(r["profit"] for r in reference_rows),
                storage_net=sum(r["storage_marginal_profit_reward"] for r in rows),
                reward=sum(r["learning_reward"] for r in rows),
                satisfaction=float(np.mean([r["mean_satisfaction"] for r in rows])),
                charge=sum(r["charge"] for r in rows), discharge=sum(r["discharge"] for r in rows),
                excess_energy=sum(r["excess_energy"] for r in rows),
                final_soc=rows[-1]["soc"], initial_soc=rows[0]["initial_soc"],
                price_mean=float(np.mean([r["price"] for r in rows])),
                price_upper_fraction=float(np.mean([r["raw_price_action"] >= 0.98 for r in rows])),
                storage_boundary_fraction=float(np.mean([abs(r["raw_storage_action"]) >= 0.98 for r in rows])),
                curves=curves)
