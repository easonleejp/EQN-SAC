"""负荷、风电、购电价的历史三源场景生成器；没有文件写入或训练副作用。

三条曲线均为 shape=(24,)，用户类别负荷为 shape=(24, 3)。每条外生源使用
独立随机流；low/medium/high 控制扰动参数，不代表真实数据校准后的置信区间。
数值沿用导出数据的单位和缩放，不能直接将利润解释为已核准的货币金额。
"""

from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from scipy.optimize import brentq

LEVELS = ("low", "medium", "high")


@dataclass
class GeneratorConfig:
    """24 个一小时时步；三个元素的参数按 LEVELS 顺序对应扰动档位。"""

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
    """围绕名义负荷/电价生成一天；风电来自相关风速和功率曲线。"""

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
                np.interp((np.arange(24) - shift) % 24, np.arange(24), self.load, period=24)
                for shift in self.shifts
            ]
        )
        # 修正负荷平移带来的均值变化，使各时刻的期望仍以名义曲线为中心。
        self.load_center_correction = self.load / (self.shift_prob @ self.shifted)
        self.mean_cf = self.cf_moment(2.0, 10.0, 1)
        self.wind_capacity = (
            self.cfg.target_penetration * self.load.mean() / self.mean_cf
        )
        self.wind_scales = {
            level: self._wind_scale_for_shape(shape)
            for level, shape in zip(LEVELS, self.cfg.wind_shapes)
        }
        self.price_corrections = {}
        for level, width in zip(LEVELS, self.cfg.spike_time_widths):
            starts = np.arange(self.t0 - width, self.t0 + width + 1)
            assert starts.min() >= 0 and starts.max() + self.cfg.spike_duration <= 24
            expected = np.ones(24)
            for start in starts:
                expected[start : start + self.cfg.spike_duration] += (self.k0 - 1) / len(starts)
            self.price_corrections[level] = expected

    def _wind_scale_for_shape(self, shape):
        """求 Weibull 尺度参数，使各风电档位具有相同的期望容量因子。"""
        def mean_capacity_factor_error(scale):
            return self.cf_moment(shape, scale, 1) - self.mean_cf

        return brentq(mean_capacity_factor_error, 3.0, 12.0)

    def cf(self, v):
        """风速 v（标量或数组）对应的容量因子，取值 [0, 1]。"""
        wind_speed = np.asarray(v)
        config = self.cfg
        return np.where(
            (wind_speed >= config.wind_cut_in) & (wind_speed < config.wind_rated_speed),
            ((wind_speed - config.wind_cut_in) / (config.wind_rated_speed - config.wind_cut_in)) ** 3,
            np.where(
                (wind_speed >= config.wind_rated_speed) & (wind_speed <= config.wind_cut_out),
                1.0, 0.0,
            ),
        )

    def cf_moment(self, k, c, power):
        """Weibull 风速下容量因子的 power 阶矩；k 为形状、c 为尺度参数。"""
        def density(wind_speed):
            return k / c * (wind_speed / c) ** (k - 1) * np.exp(-((wind_speed / c) ** k))

        def weighted_density(wind_speed):
            return float(self.cf(wind_speed)) ** power * density(wind_speed)

        # 额定风速至切出风速区间容量因子为 1，直接积分概率密度。
        return (
            quad(
                weighted_density,
                self.cfg.wind_cut_in,
                self.cfg.wind_rated_speed,
            )[0]
            + quad(density, self.cfg.wind_rated_speed, self.cfg.wind_cut_out)[0]
        )

    @staticmethod
    def ar(rng, rho):
        """生成 shape=(24,) 的单位方差 AR(1) 正态潜变量，rho 为相邻时刻相关系数。"""
        latent = rng.normal(size=24)
        for hour in range(1, 24):
            latent[hour] = rho * latent[hour - 1] + np.sqrt(1 - rho * rho) * latent[hour]
        return latent

    def sample(
        self, seed, load_level="medium", wind_level="medium", price_level="medium"
    ):
        """返回一天的外生曲线和抽样元数据；同 seed/档位重复调用得到相同结果。"""
        load_index, wind_index, price_index = [
            LEVELS.index(level) for level in (load_level, wind_level, price_level)
        ]
        load_rng, wind_rng, price_rng = [
            np.random.default_rng(child_seed)
            for child_seed in np.random.SeedSequence(int(seed)).spawn(3)
        ]
        config = self.cfg

        # 负荷 = 平移后的名义曲线 × 均值修正 × 小时扰动 × 整日扰动。
        shift_index = load_rng.choice(5, p=self.shift_prob)
        load_latent = self.ar(load_rng, config.load_rho)
        hourly_sigma = config.load_sigmas[load_index]
        daily_sigma = config.load_daily_sigmas[load_index]
        load = (
            self.shifted[shift_index]
            * self.load_center_correction
            * np.exp(hourly_sigma * load_latent - 0.5 * hourly_sigma**2)
            * np.exp(daily_sigma * load_rng.normal() - 0.5 * daily_sigma**2)
        )

        # 正态潜变量经 CDF 和 Weibull 逆 CDF 变为风速，再经过风机功率曲线。
        wind_latent = self.ar(wind_rng, config.wind_rho)
        wind_probability = np.clip(ndtr(wind_latent), 1e-12, 1 - 1e-12)
        wind_speed = self.wind_scales[wind_level] * (-np.log1p(-wind_probability)) ** (
            1 / config.wind_shapes[wind_index]
        )
        wind = self.wind_capacity * self.cf(wind_speed)

        # 脉冲除以其期望形状作均值修正；这并不保证每条最终曲线都有实际尖峰。
        spike_width = config.spike_time_widths[price_index]
        spike_start = int(price_rng.integers(self.t0 - spike_width, self.t0 + spike_width + 1))
        spike_multiplier = self.k0 + price_rng.uniform(
            -config.spike_offsets[price_index], config.spike_offsets[price_index]
        )
        pulse = np.ones(24)
        pulse[spike_start : spike_start + config.spike_duration] = spike_multiplier
        price_sigma = config.price_sigmas[price_index]
        price_latent = self.ar(price_rng, config.price_rho)
        price = (
            self.price
            * pulse
            / self.price_corrections[price_level]
            * np.exp(price_sigma * price_latent - 0.5 * price_sigma**2)
        )
        return {
            "seed": int(seed),
            "load_level": load_level,
            "wind_level": wind_level,
            "price_level": price_level,
            "load_shift_hours": float(self.shifts[shift_index]),
            "spike_time": spike_start,
            "spike_multiplier_raw": float(spike_multiplier),
            "baseline_load": load,
            "wind_power": wind,
            "wind_speed": wind_speed,
            "grid_price": price,
            "class_baseline_load": load[:, None] * self.shares,
        }

    def observations(self, r, t):
        """提取场景 r 在时刻 t 的可用观测，历史/负荷预测各含 3 个值。

        预测仅用当前负荷与已知名义曲线，比例按 rho**步长衰减；不读取未来
        实际负荷、风电或价格。单因素模式也保留这套历史预测公式。
        """
        ratio = r["baseline_load"][t] / self.load[t]
        return {
            "hour": t,
            "load_current": float(r["baseline_load"][t]),
            "wind_current": float(r["wind_power"][t]),
            "price_current": float(r["grid_price"][t]),
            "price_history": [float(r["grid_price"][max(0, t - i)]) for i in (1, 2, 3)],
            "wind_history": [float(r["wind_power"][max(0, t - i)]) for i in (1, 2, 3)],
            "load_forecast": [
                float(self.load[min(23, t + i)] * ratio ** (self.cfg.load_rho**i))
                for i in (1, 2, 3)
            ],
        }
