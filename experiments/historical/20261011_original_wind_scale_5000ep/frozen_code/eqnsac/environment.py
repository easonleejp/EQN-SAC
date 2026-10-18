"""24-hour retail/storage environment, preserving the audited historical equations.

Actions are [shared price, physical normalized storage power]; positive storage
means charging. No action replacement, optimizer, imitation, or future outcomes.
"""

import json
from types import SimpleNamespace
import numpy as np
from .config import ROOT
from .scenarios import ThreeSourceGenerator

FEATURES = (
    "price",
    "net_load",
    "temperature",
    "hour",
    "wind",
    "soc",
    "previous_price_1",
    "previous_price_2",
    "previous_price_3",
    "previous_response",
    "price_lag1",
    "price_lag2",
    "price_lag3",
    "load_forecast1",
    "load_forecast2",
    "load_forecast3",
    "wind_lag1",
    "wind_lag2",
    "wind_lag3",
)
STORAGE_INDICES = {
    "compact10": (5, 3, 0, 1, 10, 11, 12, 16, 17, 18),
    "causal15": (5, 3, 0, 1, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
}


class Environment:
    def __init__(self, storage_reward_mix=0.1):
        self.profile = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in json.loads(
                (ROOT / "data/profile.json").read_text("utf-8")
            ).items()
        }
        self.spec = json.loads((ROOT / "data/environment.json").read_text("utf-8"))
        self.generator = ThreeSourceGenerator(self.profile)
        self.storage = SimpleNamespace(**self.spec["storage"])
        b = self.storage
        if not b.soc_min <= b.soc_init <= b.soc_max or b.soc_min >= b.soc_max:
            raise ValueError("Invalid SOC limits")
        if (
            min(b.p_ch_max, b.p_dis_max, b.eta_ch, b.eta_dis) <= 0
            or max(b.eta_ch, b.eta_dis) > 1
            or b.c_es < 0
        ):
            raise ValueError("Invalid storage parameters")
        if self.spec["params"]["dt"] != 1 or b.eta_self != 1:
            raise ValueError("This audited model supports dt=1 and eta_self=1 only")
        if not 0 <= storage_reward_mix <= 1:
            raise ValueError("Reward mix must be in [0,1]")
        self.norm = self.spec["normalizers"]
        self.user = SimpleNamespace(
            **{k: np.asarray(v, dtype=np.float32) for k, v in self.spec["user"].items()}
        )
        if np.any(self.user.price_max <= self.user.price_min):
            raise ValueError("Invalid price limits")
        self.demand = SimpleNamespace(**self.spec["demand"])
        self.mix = storage_reward_mix
        self.t = 24

    def reset(self, case, response_seed=42):
        self.case = {
            k: np.asarray(v) if isinstance(v, list) else v for k, v in case.items()
        }
        self.load = np.asarray(case["baseline_load"], dtype=np.float32)
        self.wind = np.asarray(case["wind_power"], dtype=np.float32)
        self.price = np.asarray(case["grid_price"], dtype=np.float32)
        shares = self.generator.shares
        self.classes = self.load.astype(float)[:, None] * shares
        self.classes[:, 2] = self.load.astype(float) - self.classes[:, :2].sum(1)
        self.t = 0
        self.soc = self.storage.soc_init
        self.previous_response = 0.0
        self.previous_price = np.full(
            3, self.profile["retail_price"][0], dtype=np.float32
        )
        self.rng = np.random.default_rng(response_seed)
        return self.state()

    def state(self):
        if self.t >= 24:
            return np.zeros(19, dtype=np.float32)
        t = self.t
        n = self.norm
        b = self.storage
        history = lambda x, scale: [
            float(x[max(0, t - lag)]) / scale for lag in (1, 2, 3)
        ]
        forecast = self.generator.observations(self.case, t)["load_forecast"]
        prev = (self.previous_price - self.user.price_min) / (
            self.user.price_max - self.user.price_min
        )
        return np.asarray(
            [
                self.price[t] / n["max_grid_price"],
                (self.load[t] - self.wind[t]) / n["max_load"],
                self.profile["temperature"][t] / n["temp_scale"],
                t / 23,
                self.wind[t] / n["max_pv"],
                np.clip((self.soc - b.soc_min) / (b.soc_max - b.soc_min), 0, 1),
                *prev,
                self.previous_response / n["max_load"],
                *history(self.price, n["max_grid_price"]),
                *(np.asarray(forecast) / n["max_load"]),
                *history(self.wind, n["max_pv"]),
            ],
            dtype=np.float32,
        )

    def feasible_bounds(self):
        b = self.storage
        dt = self.spec["params"]["dt"]
        e = self.soc * b.eta_self
        upper = min(b.p_ch_max, max(0, (b.soc_max - e) / (b.eta_ch * dt))) / b.p_ch_max
        lower = (
            -min(b.p_dis_max, max(0, (e - b.soc_min) * b.eta_dis / dt)) / b.p_dis_max
        )
        return lower, upper

    def response(self, prices):
        t = self.t
        u = self.user
        d = self.demand
        load_g = self.classes[t].astype(np.float32)
        share = load_g / max(float(load_g.sum()), 1e-8)
        curvature = max(
            float(self.rng.normal(d.curvature_mean, d.curvature_std)), d.min_curvature
        )
        noise = float(self.rng.normal(0, d.demand_noise_std))
        reference = self.spec["hourly_demand_reference"][t]
        rational = (
            load_g
            + (prices - reference) / (2 * (-curvature / np.maximum(share, 1e-8)))
            + share * noise
        )
        theory = (load_g - rational).astype(np.float32)
        billing = np.full(3, float(self.profile["retail_price"][t]), dtype=np.float32)
        eta = np.abs(prices - billing) / np.maximum(billing, 1e-8)
        temp = np.full(
            3, 20 / (abs(self.profile["temperature"][t] - 22) + 0.001), dtype=np.float32
        )
        actual = theory.copy()
        for _ in range(3):
            comfort = 1 - np.square(actual) / np.maximum(np.square(load_g), 1e-8)
            bill = prices * np.maximum(load_g - actual, 0)
            base = billing * load_g
            economic = 1 - (bill - base) / np.maximum(np.abs(base), 1e-8)
            satisfaction = np.clip(
                u.comfort_weight * comfort + u.comp_weight * economic, 0, 5
            )
            participation = 1 / (1 + np.exp(-u.lambda_s * (satisfaction - u.s_ref)))
            mu_p = 1 / (1 + np.exp(-4.2 * (eta - 0.05)))
            mu_t = 1 / (1 + np.exp(-(temp - 1)))
            mu_s = 1 / (1 + np.exp(-4 * (participation - 0.5)))
            mu = np.clip(
                0.42 * mu_p + 0.22 * mu_t + 0.36 * mu_s + self.rng.normal(0, 0.025, 3),
                0,
                1,
            )
            actual = np.minimum(mu * theory, load_g)
        # Historical satisfaction is from the last fixed-point iteration input.
        return actual.astype(np.float32), load_g, float(satisfaction.mean())

    def step(self, action):
        if self.t >= 24:
            raise RuntimeError("Reset before stepping a terminated episode")
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("Expected two finite actions")
        a = np.clip(a, -1, 1)
        t = self.t
        b = self.storage
        dt = self.spec["params"]["dt"]
        prices = self.user.price_min + (a[0] + 1) * 0.5 * (
            self.user.price_max - self.user.price_min
        )
        actual, classes, satisfaction = self.response(prices)
        user_load = float(np.sum(classes - actual))
        total_response = float(actual.sum())
        lo, hi = self.feasible_bounds()
        executed = min(hi, max(lo, float(a[1])))
        charge = max(executed, 0) * b.p_ch_max
        discharge = max(-executed, 0) * b.p_dis_max
        self.soc = (
            self.soc * b.eta_self + (b.eta_ch * charge - discharge / b.eta_dis) * dt
        )
        if not b.soc_min - 1e-6 <= self.soc <= b.soc_max + 1e-6:
            raise RuntimeError("SOC violation")
        self.soc = float(np.clip(self.soc, b.soc_min, b.soc_max))
        cost = b.c_es * (charge + discharge) * dt
        purchase = max(user_load + charge - discharge - float(self.wind[t]), 0)
        grid = float(self.price[t] * purchase * dt)
        revenue = float(np.sum(prices * (classes - actual)))
        profit = revenue - grid - cost
        no_storage = float(self.price[t] * max(user_load - float(self.wind[t]), 0) * dt)
        marginal = no_storage - grid - cost
        mixed = self.mix * profit + (1 - self.mix) * marginal
        self.previous_price = prices.astype(np.float32)
        self.previous_response = total_response
        self.t += 1
        info = dict(
            profit=profit,
            storage_marginal_profit_reward=marginal,
            storage_training_reward=mixed,
            soc=self.soc,
            charge=charge,
            discharge=discharge,
            user_load=user_load,
            mean_satisfaction=satisfaction,
            total_response=total_response,
            grid_cost=grid,
            revenue=revenue,
            storage_cost=cost,
            executed_storage_action=executed,
            price=float(prices.mean()),
            hour=t,
        )
        return (
            self.state(),
            np.asarray([profit, mixed], dtype=np.float32),
            self.t == 24,
            info,
        )
