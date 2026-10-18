"""Historical three-source generator; no filesystem or training side effects."""

from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from scipy.optimize import brentq

LEVELS = ("low", "medium", "high")


@dataclass
class GeneratorConfig:
    horizon: int = 24
    dt: float = 1.0
    target_penetration: float = 0.2
    load_rho: float = 0.85
    wind_rho: float = 0.9
    price_rho: float = 0.8
    wind_cut_in: float = 3.0
    wind_rated_speed: float = 12.0
    wind_cut_out: float = 25.0
    load_sigmas: tuple = (0.04, 0.09, 0.16)
    load_daily_sigmas: tuple = (0.02, 0.05, 0.09)
    wind_shapes: tuple = (7.0, 3.5, 2.0)
    price_sigmas: tuple = (0.025, 0.07, 0.14)
    spike_offsets: tuple = (0.05, 0.25, 0.6)
    spike_time_widths: tuple = (0, 1, 2)
    spike_duration: int = 3
    seed: int = 20260914


class ThreeSourceGenerator:

    def __init__(self, profile, cfg=None, spike_time=17, spike_multiplier=2.0):
        self.cfg = cfg or GeneratorConfig()
        self.profile = {k: np.asarray(v).copy() for k, v in profile.items()}
        assert (
            self.cfg.horizon == 24 and self.cfg.dt == 1.0
        ), "Preview preserves original hourly horizon"
        self.load = np.asarray(profile["baseline_load"], float)
        self.price = np.asarray(profile["grid_price"], float)
        self.shares = (
            np.asarray(profile["class_baseline_load"], float) / self.load[:, None]
        )
        self.t0 = int(spike_time)
        self.k0 = float(spike_multiplier)
        self.shifts = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.shift_prob = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        self.shifted = np.array(
            [
                np.interp((np.arange(24) - s) % 24, np.arange(24), self.load, period=24)
                for s in self.shifts
            ]
        )
        self.load_center_correction = self.load / (self.shift_prob @ self.shifted)
        self.mean_cf = self.cf_moment(2.0, 10.0, 1)
        self.wind_capacity = (
            self.cfg.target_penetration * self.load.mean() / self.mean_cf
        )
        self.wind_scales = {
            level: brentq(lambda c: self.cf_moment(k, c, 1) - self.mean_cf, 3.0, 12.0)
            for level, k in zip(LEVELS, self.cfg.wind_shapes)
        }
        self.price_corrections = {}
        for level, width in zip(LEVELS, self.cfg.spike_time_widths):
            starts = np.arange(self.t0 - width, self.t0 + width + 1)
            assert starts.min() >= 0 and starts.max() + self.cfg.spike_duration <= 24
            expected = np.ones(24)
            for t in starts:
                expected[t : t + self.cfg.spike_duration] += (self.k0 - 1) / len(starts)
            self.price_corrections[level] = expected

    def cf(self, v):
        v = np.asarray(v)
        c = self.cfg
        return np.where(
            (v >= c.wind_cut_in) & (v < c.wind_rated_speed),
            ((v - c.wind_cut_in) / (c.wind_rated_speed - c.wind_cut_in)) ** 3,
            np.where((v >= c.wind_rated_speed) & (v <= c.wind_cut_out), 1.0, 0.0),
        )

    def cf_moment(self, k, c, power):

        def density(v):
            return k / c * (v / c) ** (k - 1) * np.exp(-((v / c) ** k))

        return (
            quad(
                lambda v: float(self.cf(v)) ** power * density(v),
                self.cfg.wind_cut_in,
                self.cfg.wind_rated_speed,
            )[0]
            + quad(density, self.cfg.wind_rated_speed, self.cfg.wind_cut_out)[0]
        )

    @staticmethod
    def ar(rng, rho):
        z = rng.normal(size=24)
        for t in range(1, 24):
            z[t] = rho * z[t - 1] + np.sqrt(1 - rho * rho) * z[t]
        return z

    def sample(
        self, seed, load_level="medium", wind_level="medium", price_level="medium"
    ):
        indices = [LEVELS.index(x) for x in (load_level, wind_level, price_level)]
        lr, wr, pr = [
            np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(3)
        ]
        il, iw, ip = indices
        c = self.cfg
        shift_idx = lr.choice(5, p=self.shift_prob)
        z = self.ar(lr, c.load_rho)
        sd = c.load_sigmas[il]
        dd = c.load_daily_sigmas[il]
        load = (
            self.shifted[shift_idx]
            * self.load_center_correction
            * np.exp(sd * z - 0.5 * sd**2)
            * np.exp(dd * lr.normal() - 0.5 * dd**2)
        )
        wz = self.ar(wr, c.wind_rho)
        u = np.clip(ndtr(wz), 1e-12, 1 - 1e-12)
        v = self.wind_scales[wind_level] * (-np.log1p(-u)) ** (1 / c.wind_shapes[iw])
        wind = self.wind_capacity * self.cf(v)
        width = c.spike_time_widths[ip]
        t = int(pr.integers(self.t0 - width, self.t0 + width + 1))
        k = self.k0 + pr.uniform(-c.spike_offsets[ip], c.spike_offsets[ip])
        pulse = np.ones(24)
        pulse[t : t + c.spike_duration] = k
        ps = c.price_sigmas[ip]
        pz = self.ar(pr, c.price_rho)
        price = (
            self.price
            * pulse
            / self.price_corrections[price_level]
            * np.exp(ps * pz - 0.5 * ps**2)
        )
        return {
            "seed": int(seed),
            "load_level": load_level,
            "wind_level": wind_level,
            "price_level": price_level,
            "load_shift_hours": float(self.shifts[shift_idx]),
            "spike_time": t,
            "spike_multiplier_raw": float(k),
            "baseline_load": load,
            "wind_power": wind,
            "wind_speed": v,
            "grid_price": price,
            "class_baseline_load": load[:, None] * self.shares,
        }

    def observations(self, r, t):
        ratio = r["baseline_load"][t] / self.load[t]
        return {
            "hour": t,
            "load_current": float(r["baseline_load"][t]),
            "wind_current": float(r["wind_power"][t]),
            "price_current": float(r["grid_price"][t]),
            "price_history": [float(r["grid_price"][max(0, t - i)]) for i in (1, 2, 3)],
            "wind_history": [float(r["wind_power"][max(0, t - i)]) for i in (1, 2, 3)],
            "load_forecast": [
                float(self.load[min(23, t + i)] * ratio**self.cfg.load_rho**i)
                for i in (1, 2, 3)
            ],
        }
