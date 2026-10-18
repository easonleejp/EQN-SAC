"""24 小时零售定价与储能环境，保留历史环境的响应和结算公式。

动作 shape=(2,)：[共享归一化零售价，物理归一化储能功率]。
储能动作正值充电、负值放电；这里的 soc 是储能量，状态中的 soc 才是归一化比例。
时步固定为 1 小时。功率、能量和价格沿用导出数据，货币/电价单位尚待核准。
"""

import copy
import json
from collections.abc import Mapping
from types import SimpleNamespace
import numpy as np
from .config import ROOT
from .scenarios import ThreeSourceGenerator

# 顺序是模型、回放和历史检查点的共同约定，调整排列会改变模型输入含义。
# previous_price_1/2/3 指上小时三类用户的零售价，并非零售价的三个时间滞后。
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
STATE_DIM = len(FEATURES)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}
ACTION_NAMES = ("price", "storage")
ACTION_DIM = len(ACTION_NAMES)
PRICE_INDICES = tuple(
    FEATURE_INDEX[name] for name in ("price", "net_load", "temperature", "hour")
)
STORAGE_INDICES = {
    "compact10": tuple(
        FEATURE_INDEX[name]
        for name in (
            "soc", "hour", "price", "net_load",
            "price_lag1", "price_lag2", "price_lag3",
            "wind_lag1", "wind_lag2", "wind_lag3",
        )
    ),
    "causal15": tuple(
        FEATURE_INDEX[name]
        for name in (
            "soc", "hour", "price", "net_load", "wind", "previous_response",
            "price_lag1", "price_lag2", "price_lag3",
            "load_forecast1", "load_forecast2", "load_forecast3",
            "wind_lag1", "wind_lag2", "wind_lag3",
        )
    ),
}


class Environment:
    """一个 episode 对应一天；reset 提供外生场景，step 推进一步响应与结算。"""

    def __init__(self, storage_reward_mix=0.1, *, spec=None, generator_config=None):
        self.profile = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in json.loads(
                (ROOT / "data/profile.json").read_text("utf-8")
            ).items()
        }
        self.spec = (json.loads((ROOT / "data/environment.json").read_text("utf-8"))
                     if spec is None else copy.deepcopy(spec))
        self.generator = ThreeSourceGenerator(self.profile, generator_config)
        self.storage = SimpleNamespace(**self.spec["storage"])
        storage = self.storage
        if (
            not storage.soc_min <= storage.soc_init <= storage.soc_max
            or storage.soc_min >= storage.soc_max
        ):
            raise ValueError("Invalid SOC limits")
        if (
            min(storage.p_ch_max, storage.p_dis_max, storage.eta_ch, storage.eta_dis) <= 0
            or max(storage.eta_ch, storage.eta_dis) > 1
            or storage.c_es < 0
        ):
            raise ValueError("Invalid storage parameters")
        if self.spec["params"]["dt"] != 1 or storage.eta_self != 1:
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
        """复制三条 shape=(24,) 的外生曲线，返回首小时 shape=(STATE_DIM,) 状态。

        用户类别负荷按名义占比分配，shape=(24, 3)。先完成校验，再替换活动
        episode；外部继续修改 case 不会改变本次模拟的预测或结算。
        """
        if not isinstance(case, Mapping):
            raise ValueError("Scenario must be a mapping")
        snapshot = copy.deepcopy(dict(case))
        arrays = {}
        for key in ("baseline_load", "wind_power", "grid_price"):
            if key not in snapshot:
                raise ValueError("Scenario is missing " + key)
            try:
                source = np.array(snapshot[key], dtype=np.float64, copy=True)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(key + " must contain 24 finite numbers") from exc
            if source.shape != (24,) or not np.isfinite(source).all():
                raise ValueError(key + " must contain 24 finite numbers")
            if key != "grid_price" and np.any(source < 0):
                raise ValueError(key + " must be nonnegative")
            with np.errstate(over="ignore", invalid="ignore"):
                values = source.astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError(key + " exceeds the float32 range")
            snapshot[key] = source
            arrays[key] = values
        # RNG 初始化也可能失败，必须放在替换活动状态之前。
        rng = np.random.default_rng(response_seed)
        shares = self.generator.shares
        load = arrays["baseline_load"]
        classes = load.astype(float)[:, None] * shares
        # 第三类取余量，保持三类之和等于本次场景总负荷。
        classes[:, 2] = load.astype(float) - classes[:, :2].sum(1)
        self.case = snapshot
        self.load = load
        self.wind = arrays["wind_power"]
        self.price = arrays["grid_price"]
        self.classes = classes
        self.t = 0
        self.soc = self.storage.soc_init
        self.previous_response = 0.0
        self.previous_price = np.full(
            3, self.profile["retail_price"][0], dtype=np.float32
        )
        self.rng = rng
        return self.state()

    def _normalized_history(self, values, scale):
        """前三小时的已知量；日初不足的历史位置用第 0 小时填充。"""
        return [float(values[max(0, self.t - lag)]) / scale for lag in (1, 2, 3)]

    def state(self):
        """按 FEATURES 顺序生成 float32 状态；终止状态为同维度零向量。"""
        if self.t >= 24:
            return np.zeros(STATE_DIM, dtype=np.float32)
        hour = self.t
        normalizers = self.norm
        storage = self.storage
        forecast = self.generator.observations(self.case, hour)["load_forecast"]
        previous_prices = (self.previous_price - self.user.price_min) / (
            self.user.price_max - self.user.price_min
        )
        return np.asarray(
            [
                self.price[hour] / normalizers["max_grid_price"],
                (self.load[hour] - self.wind[hour]) / normalizers["max_load"],
                self.profile["temperature"][hour] / normalizers["temp_scale"],
                hour / 23,
                self.wind[hour] / normalizers["max_pv"],  # 历史字段名，实际用于风电。
                np.clip(
                    (self.soc - storage.soc_min) / (storage.soc_max - storage.soc_min),
                    0, 1,
                ),
                *previous_prices,
                self.previous_response / normalizers["max_load"],
                *self._normalized_history(self.price, normalizers["max_grid_price"]),
                *(np.asarray(forecast) / normalizers["max_load"]),
                *self._normalized_history(self.wind, normalizers["max_pv"]),
            ],
            dtype=np.float32,
        )

    def feasible_bounds(self):
        """返回储能物理归一化动作的 (放电下界, 充电上界)，均在 [-1, 1]。"""
        storage = self.storage
        dt = self.spec["params"]["dt"]
        energy_after_self_discharge = self.soc * storage.eta_self
        upper = min(
            storage.p_ch_max,
            max(0, (storage.soc_max - energy_after_self_discharge) / (storage.eta_ch * dt)),
        ) / storage.p_ch_max
        lower = (
            -min(
                storage.p_dis_max,
                max(0, (energy_after_self_discharge - storage.soc_min) * storage.eta_dis / dt),
            ) / storage.p_dis_max
        )
        return lower, upper

    def response(self, prices):
        """将三类用户价格 shape=(3,) 转为实际削减量、基线负荷和平均满意度。

        正响应表示削减用电，负响应表示增加用电。每小时先抽曲率和需求噪声，
        再做三次满意度—参与响应迭代；随机调用次数和顺序与历史模型一致。
        """
        hour = self.t
        user = self.user
        demand = self.demand
        class_load = self.classes[hour].astype(np.float32)
        class_share = class_load / max(float(class_load.sum()), 1e-8)
        curvature = max(
            float(self.rng.normal(demand.curvature_mean, demand.curvature_std)),
            demand.min_curvature,
        )
        demand_noise = float(self.rng.normal(0, demand.demand_noise_std))
        reference_price = self.spec["hourly_demand_reference"][hour]
        rational_load = (
            class_load
            + (prices - reference_price) / (2 * (-curvature / np.maximum(class_share, 1e-8)))
            + class_share * demand_noise
        )
        theoretical_response = (class_load - rational_load).astype(np.float32)
        billing_price = np.full(
            3, float(self.profile["retail_price"][hour]), dtype=np.float32
        )
        relative_price_change = np.abs(prices - billing_price) / np.maximum(billing_price, 1e-8)
        temperature_sensitivity = np.full(
            3, 20 / (abs(self.profile["temperature"][hour] - 22) + 0.001), dtype=np.float32
        )
        actual_response = theoretical_response.copy()
        for _ in range(3):
            comfort = 1 - np.square(actual_response) / np.maximum(np.square(class_load), 1e-8)
            bill = prices * np.maximum(class_load - actual_response, 0)
            baseline_bill = billing_price * class_load
            economic = 1 - (bill - baseline_bill) / np.maximum(np.abs(baseline_bill), 1e-8)
            satisfaction = np.clip(
                user.comfort_weight * comfort + user.comp_weight * economic, 0, 5
            )
            participation = 1 / (1 + np.exp(-user.lambda_s * (satisfaction - user.s_ref)))
            price_factor = 1 / (1 + np.exp(-4.2 * (relative_price_change - 0.05)))
            temperature_factor = 1 / (1 + np.exp(-(temperature_sensitivity - 1)))
            satisfaction_factor = 1 / (1 + np.exp(-4 * (participation - 0.5)))
            response_fraction = np.clip(
                0.42 * price_factor
                + 0.22 * temperature_factor
                + 0.36 * satisfaction_factor
                + self.rng.normal(0, 0.025, 3),
                0, 1,
            )
            # 削减量不超过该类负荷，保证响应后的购电需求非负。
            actual_response = np.minimum(response_fraction * theoretical_response, class_load)
        # 历史满意度取最后一次迭代的输入响应计算值；不额外重算或消耗随机数。
        return actual_response.astype(np.float32), class_load, float(satisfaction.mean())

    def step(self, action):
        """推进一小时，返回 (19 维状态, 2 维奖励, 是否终止, 结算明细)。

        奖励顺序为 [零售总利润, 储能训练奖励]。这里只执行 SOC/功率物理限制，
        不用优化器或未来实现替换策略动作。日末 SOC 自由，没有残值奖励。
        """
        if self.t >= 24:
            raise RuntimeError("Reset before stepping a terminated episode")
        normalized_action = np.asarray(action, dtype=np.float32)
        if normalized_action.shape != (ACTION_DIM,) or not np.isfinite(normalized_action).all():
            raise ValueError("Expected two finite actions")
        normalized_action = np.clip(normalized_action, -1, 1)
        hour = self.t
        storage = self.storage
        dt = self.spec["params"]["dt"]

        # 同一个归一化价格动作映射到三类用户各自的价格范围。
        prices = self.user.price_min + (normalized_action[0] + 1) * 0.5 * (
            self.user.price_max - self.user.price_min
        )
        actual_response, class_load, satisfaction = self.response(prices)
        user_load = float(np.sum(class_load - actual_response))
        total_response = float(actual_response.sum())
        lower, upper = self.feasible_bounds()
        executed = min(upper, max(lower, float(normalized_action[1])))
        charge = max(executed, 0) * storage.p_ch_max
        discharge = max(-executed, 0) * storage.p_dis_max

        # 储能量平衡：充电只有 eta_ch 部分入库，放电需从库中扣除 discharge/eta_dis。
        self.soc = (
            self.soc * storage.eta_self
            + (storage.eta_ch * charge - discharge / storage.eta_dis) * dt
        )
        if not storage.soc_min - 1e-6 <= self.soc <= storage.soc_max + 1e-6:
            raise RuntimeError("SOC violation")
        self.soc = float(np.clip(self.soc, storage.soc_min, storage.soc_max))

        # 不向电网售电：风电和放电抵扣需求后，多余电量不产生销售收入。
        storage_cost = storage.c_es * (charge + discharge) * dt
        grid_purchase = max(user_load + charge - discharge - float(self.wind[hour]), 0)
        grid_cost = float(self.price[hour] * grid_purchase * dt)
        revenue = float(np.sum(prices * (class_load - actual_response)))
        profit = revenue - grid_cost - storage_cost
        grid_cost_without_storage = float(
            self.price[hour] * max(user_load - float(self.wind[hour]), 0) * dt
        )
        storage_marginal = grid_cost_without_storage - grid_cost - storage_cost
        # 混合奖励 = mix×总利润 + (1-mix)×储能边际收益。
        # 等价于 mix×无储能零售利润 + 储能边际收益；不改变原计算顺序。
        storage_reward = self.mix * profit + (1 - self.mix) * storage_marginal
        self.previous_price = prices.astype(np.float32)
        self.previous_response = total_response
        self.t += 1
        info = dict(
            profit=profit,
            storage_marginal_profit_reward=storage_marginal,
            storage_training_reward=storage_reward,
            soc=self.soc,
            charge=charge,
            discharge=discharge,
            user_load=user_load,
            mean_satisfaction=satisfaction,
            total_response=total_response,
            grid_cost=grid_cost,
            revenue=revenue,
            storage_cost=storage_cost,
            executed_storage_action=executed,
            price=float(prices.mean()),
            hour=hour,
        )
        return (
            self.state(),
            np.asarray([profit, storage_reward], dtype=np.float32),
            self.t == 24,
            info,
        )
