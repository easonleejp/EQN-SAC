"""单因素场景：只给选中的外生源乘一个整日系数，其他输入保持固定。"""

import numpy as np


def sample_case(env, cfg, seed, levels=("medium", "medium", "medium")):
    """按配置生成一天；曲线 shape=(24,)，用户类别负荷 shape=(24, 3)。

    three_source 使用 levels 指定三个源的扰动档位。单因素模式抽取一个
    delta ~ Uniform(-bound, bound)，将对应整条曲线乘以 1+delta，因此日内
    形状保持不变；响应随机流是否固定由调用方根据 cfg.response_rng 处理。
    """
    if cfg.scenario_mode == "three_source":
        return env.generator.sample(seed, *levels)

    # 导出的名义输入没有风电曲线：固定 reference_seed 的 low 档样本作为参照。
    # 每次重新生成相同参照，避免调用方修改返回值后污染后续案例。
    reference = env.generator.sample(cfg.reference_seed, "low", "low", "low")
    if cfg.scenario_mode not in ("load_scale", "wind_scale", "price_scale"):
        raise ValueError("Unsupported single-source mode")
    bound = {
        "load_scale": cfg.load_scale_bound,
        "wind_scale": cfg.wind_scale_bound,
        "price_scale": cfg.price_scale_bound,
    }[cfg.scenario_mode]
    delta = float(np.random.default_rng(seed).uniform(-bound, bound))
    load_delta = delta if cfg.scenario_mode == "load_scale" else 0.0
    wind_delta = delta if cfg.scenario_mode == "wind_scale" else 0.0
    price_delta = delta if cfg.scenario_mode == "price_scale" else 0.0
    load = env.generator.load * (1 + load_delta)
    if cfg.scenario_mode == "wind_scale":
        # 用整个允许范围的上界检查装机容量；截断会破坏“同一曲线整日缩放”。
        if reference["wind_power"].max() * (1 + bound) > env.generator.wind_capacity:
            raise ValueError("Wind scaling exceeds capacity; do not silently clip curve shape")
        reference["wind_power"] = reference["wind_power"] * (1 + wind_delta)
        # 调整的是风电功率，原风速仅作参照，不能解释为调整后功率对应的风速。
        reference["reference_wind_speed"] = reference.pop("wind_speed")
    reference.update(
        seed=int(seed),
        scenario_mode=cfg.scenario_mode,
        load_delta=load_delta,
        wind_delta=wind_delta,
        price_delta=price_delta,
        reference_seed=cfg.reference_seed,
        load_level="daily_uniform" if cfg.scenario_mode == "load_scale" else "nominal_fixed",
        wind_level="daily_uniform" if cfg.scenario_mode == "wind_scale" else "fixed",
        price_level="daily_uniform" if cfg.scenario_mode == "price_scale" else "nominal_fixed",
        load_shift_hours=0.0,
        spike_time=-1,
        spike_multiplier_raw=1.0,
        baseline_load=load,
        class_baseline_load=load[:, None] * env.generator.shares,
        grid_price=env.generator.price * (1 + price_delta),
    )
    return reference
