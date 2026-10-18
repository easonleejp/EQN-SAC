"""绘制固定同一天的 500 回合训练及一个已保存的真实测试回合。

只读 training.csv / test_1ep.json 等日志；不导入模型、不执行策略或测试。
仅训练利润叠加 20 回合尾随均值，24 小时测试值不平均、不平滑。

Example:
    .venv/bin/python tools/plot_sac_fixed_day.py \
        --run-dir experiments/sac_causal13_500_20261018/run \
        --output experiments/sac_causal13_500_20261018/figures
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import StrMethodFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GRAY, GREEN = "#24639b", "#cc6c2d", "#747e89", "#287e6d"
INPUT_NAMES = (
    "config.json", "complete.json", "manifest.json", "training.csv",
    "test_1ep.json", "effective_environment.json", "effective_profile.json",
)
HOURLY_KEYS = (
    "hour", "grid_price", "price", "charge", "discharge", "initial_soc", "soc",
    "baseline_load", "user_load", "wind", "grid_purchase", "profit",
    "total_response", "learning_reward", "reference_profit",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_inputs(run_dir):
    """校验固定日、完整 500 回合、单测试回合和基本能量会计。"""
    inputs = [fingerprint(run_dir / name) for name in INPUT_NAMES]
    data = {name: json.loads((run_dir / name).read_text("utf-8"))
            for name in INPUT_NAMES if name.endswith(".json")}
    cfg, complete = data["config.json"], data["complete.json"]
    result = data["test_1ep.json"]
    env, profile = data["effective_environment.json"], data["effective_profile.json"]
    require(cfg["episodes"] == complete["episodes"] == 500,
            "本图只接受已完成的 500 回合训练。")
    require(cfg["scenario_mode"] == "deterministic_day", "必须使用固定同一天场景。")
    noise_keys = ("curvature_std", "demand_noise_std", "participation_noise_std")
    require(all(env["demand"][key] == 0 for key in noise_keys),
            "环境曲率、需求及参与噪声必须全部为 0。")
    require(result["n"] == len(result["cases"]) == 1 and result["deterministic"] is True,
            "test_1ep.json 必须恰好包含 1 个确定性动作测试回合。")
    for key, expected in (("config", cfg), ("environment", env), ("profile", profile)):
        require(result["metadata"][key] == expected, f"测试与训练保存的 {key} 不一致。")

    with (run_dir / "training.csv").open(encoding="utf-8", newline="") as stream:
        training = list(csv.DictReader(stream))
    episodes = np.asarray([int(row["episode"]) for row in training])
    profits = np.asarray([float(row["profit"]) for row in training])
    require(np.array_equal(episodes, np.arange(1, 501)), "训练日志必须完整包含回合 1—500。")
    require(np.isfinite(profits).all(), "训练利润含非有限值。")

    case = result["cases"][0]
    rows = case["hourly"]
    require(len(rows) == 24, "单次测试必须恰好包含 24 个小时。")
    values = {key: np.asarray([row[key] for row in rows], dtype=float) for key in HOURLY_KEYS}
    require(all(np.isfinite(array).all() for array in values.values()), "测试数据含非有限值。")
    require(np.array_equal(values["hour"], np.arange(24)), "测试小时必须按 0—23 连续排列。")
    for key in ("baseline_load", "grid_price"):
        require(np.array_equal(values[key], np.asarray(profile[key])),
                f"固定日 {key} 与保存的原始 profile 不一致。")
    require(np.all(values["wind"] == values["wind"][0]), "本轮固定日应为恒定风电功率。")

    storage = env["storage"]
    require(np.all(values["charge"] >= 0) and np.all(values["discharge"] >= 0),
            "充放电记录应分别为非负物理功率。")
    require(np.all(values["charge"] <= storage["p_ch_max"] + 1e-5)
            and np.all(values["discharge"] <= storage["p_dis_max"] + 1e-5),
            "实际充放电功率超出保存的物理上限。")
    energy = np.r_[values["initial_soc"][0], values["soc"]]
    require(np.all(energy >= storage["soc_min"] - 1e-5)
            and np.all(energy <= storage["soc_max"] + 1e-5), "SOC 超出物理范围。")
    require(np.allclose(values["initial_soc"][1:], values["soc"][:-1], rtol=0, atol=1e-5),
            "相邻小时的 SOC 不连续。")
    net_power = values["charge"] - values["discharge"]
    calculated_grid = np.maximum(values["user_load"] - values["wind"] + net_power, 0)
    require(np.allclose(calculated_grid, values["grid_purchase"], rtol=0, atol=1e-5),
            "购电记录与负荷、风电、实际充放电不满足能量平衡。")
    require(np.isclose(values["profit"].sum(), case["profit"], rtol=0, atol=1e-5),
            "逐小时利润之和与回合利润不一致。")
    return inputs, cfg, env, profile, episodes, profits, values, energy


def set_style(font):
    font_path = font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font],
        "axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11, "legend.fontsize": 9, "axes.titlepad": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    return font_path


def tidy(ax, title, ylabel, hourly=True):
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis="y", alpha=.17, linewidth=.8)
    if hourly:
        ax.set(xlabel="小时", xticks=np.arange(0, 25, 3), xlim=(0, 24))
    ax.legend(loc="best", frameon=False)


def hourly_line(ax, values, label, color, **kwargs):
    """每小时分段常数；不在相邻小时之间制造平滑过渡。"""
    ax.stairs(values, np.arange(25), label=label, color=color,
              linewidth=1.9, baseline=None, **kwargs)


def plot_results(output, cfg, env, profile, episodes, profits, values, energy):
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle("固定同一天｜SAC 训练与单回合测试", x=.07, ha="left", y=.973,
                 fontsize=21, weight="bold")
    fig.text(.07, .938, f"500 回合训练 · 1 回合测试 · 原利润目标 · seed {cfg['seed']}",
             fontsize=12, color="#475467")

    ax = axes[0, 0]
    ax.plot(episodes, profits, color=GRAY, linewidth=.8, alpha=.55, label="逐回合原利润")
    # 第 20 回合才开始：每一点只使用它及之前 19 回合，无未来信息。
    cumulative = np.r_[0., np.cumsum(profits)]
    trailing_mean = (cumulative[20:] - cumulative[:-20]) / 20
    ax.plot(episodes[19:], trailing_mean, color=BLUE, linewidth=2, label="20 回合尾随均值")
    tidy(ax, "a  训练利润（未乘奖励缩放系数）", "每日利润（澳分）", hourly=False)
    ax.set(xlabel="训练回合", xlim=(1, 500))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    ax = axes[0, 1]
    hourly_line(ax, profile["retail_price"], "原始 TOU 零售价", ORANGE, linestyle="--")
    hourly_line(ax, values["grid_price"], "批发电价", GRAY)
    hourly_line(ax, values["price"], "SAC 零售价", BLUE)
    tidy(ax, "b  单日零售价与批发价", "电价（澳分/kWh）")

    ax = axes[1, 0]
    hours = values["hour"]
    ax.bar(hours + .5, values["charge"], width=.8, color=BLUE, label="充电（正）")
    ax.bar(hours + .5, -values["discharge"], width=.8, color=ORANGE, label="放电（负）")
    ax.axhline(0, color=GRAY, linewidth=.8)
    tidy(ax, "c  实际执行的储能功率", "功率（kW）")

    ax = axes[1, 1]
    storage = env["storage"]
    ax.axhspan(storage["soc_min"], storage["soc_max"], color=BLUE, alpha=.04)
    ax.axhline(storage["soc_max"], color=GRAY, linewidth=1, linestyle="--", label="物理上、下限")
    ax.axhline(storage["soc_min"], color=GRAY, linewidth=1, linestyle="--")
    ax.plot(np.arange(25), energy, color=BLUE, linewidth=1.9, marker="o", markersize=3,
            label="小时边界的储能量")
    tidy(ax, "d  SOC（以储能量表示）", "储能量（kWh）")
    ax.set_xlabel("小时边界（0 为初始状态）")
    span = storage["soc_max"] - storage["soc_min"]
    ax.set_ylim(storage["soc_min"] - .09 * span, storage["soc_max"] + .15 * span)

    ax = axes[2, 0]
    hourly_line(ax, values["baseline_load"], "原负荷", GRAY, linestyle="--")
    hourly_line(ax, values["user_load"], "价格响应后负荷", BLUE)
    tidy(ax, "e  用户负荷：价格响应前后", "功率（kW）")

    ax = axes[2, 1]
    hourly_line(ax, np.maximum(values["baseline_load"] - values["wind"], 0),
                "原负荷扣风电", GRAY, linestyle="--")
    hourly_line(ax, np.maximum(values["user_load"] - values["wind"], 0),
                "同响应、零储能", ORANGE, linestyle=":")
    hourly_line(ax, values["grid_purchase"], "实际购电", BLUE)
    tidy(ax, "f  电网购电：分离响应与储能影响", "功率（kW）")

    footer = (
        "固定同一 24 小时场景训练 500 回合后，仅测试 1 回合；曲率、需求、参与三项环境噪声均为 0，SAC 训练探索仍有随机性。\n"
        "测试值来自单个真实回合，无跨案例平均、无动作或负荷平滑；仅训练利润叠加尾随均值。购电曲线均取非负值；“同响应、零储能”为会计分解。\n"
        "单位采用本轮建模约定：电价澳分/kWh、功率 kW、能量 kWh、利润澳分；原始数据来源未标明单位。SOC 点间连线仅辅助阅读。"
    )
    fig.text(.07, .025, footer, fontsize=9.2, color="#475467", linespacing=1.65, va="bottom")
    fig.subplots_adjust(left=.09, right=.98, top=.885, bottom=.14, hspace=.48, wspace=.29)
    paths = [output / "fixed_day_results.png", output / "fixed_day_results.pdf"]
    fig.savefig(paths[0], dpi=220)
    fig.savefig(paths[1], metadata={"Title": "固定同一天 SAC：500 回合训练与 1 回合测试",
                                  "CreationDate": None, "ModDate": None})
    plt.close(fig)
    return paths


def write_hourly(path, env, profile, values):
    derived = {
        "tou_price": np.asarray(profile["retail_price"]),
        "signed_storage_power": values["charge"] - values["discharge"],
        "baseline_grid_without_response_without_storage": np.maximum(values["baseline_load"] - values["wind"], 0),
        "response_grid_without_storage": np.maximum(values["user_load"] - values["wind"], 0),
        "soc_min": np.full(24, env["storage"]["soc_min"]),
        "soc_max": np.full(24, env["storage"]["soc_max"]),
    }
    columns = {**values, **derived}
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for hour in range(24):
            row = {key: float(array[hour]) for key, array in columns.items()}
            row["hour"] = hour
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "experiments/sac_causal13_500_20261018/run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", default="PingFang SC")
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    inputs, cfg, env, profile, episodes, profits, values, energy = read_inputs(run_dir)
    script = fingerprint(__file__)
    font_path = set_style(args.font)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = plot_results(output, cfg, env, profile, episodes, profits, values, energy)
    hourly_path = output / "hourly.csv"
    write_hourly(hourly_path, env, profile, values)
    artifacts.append(hourly_path)
    require(inputs == [fingerprint(run_dir / name) for name in INPUT_NAMES],
            "绘图期间输入文件改变，请在训练和测试写入结束后重试。")
    manifest = {
        "purpose": "固定同一天：500 回合训练及一个已保存测试回合的只读绘图",
        "training_episodes": 500, "test_episodes": 1, "test_hours": 24,
        "policy_loaded_or_evaluated": False,
        "test_aggregation": "none", "test_smoothing": "none",
        "training_smoothing": "20-episode trailing arithmetic mean, begins at episode 20",
        "grid_decomposition": "All import curves use max(load - wind + charge - discharge, 0); zero-storage curve holds actual response fixed.",
        "unit_assumption": {"price": "Australian cents/kWh", "power": "kW",
                            "energy": "kWh", "profit": "Australian cents",
                            "source_units_verified": False},
        "environment_noise": {key: env["demand"][key] for key in
                              ("curvature_std", "demand_noise_std", "participation_noise_std")},
        "inputs": inputs, "script": script, "outputs": [fingerprint(path) for path in artifacts],
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__, "font": args.font, "font_path": font_path},
        "manifest_note": "This manifest records inputs/script/three artifacts; it does not hash itself.",
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps({"outputs": [str(path) for path in artifacts], "manifest": str(manifest_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
