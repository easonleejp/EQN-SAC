"""只读比较两次固定日 SAC：各 500 回合训练、各一个已保存测试回合。

不导入 SAC 模型，不训练、不评价策略；测试的 24 个小时不平均、不平滑。
仅训练利润叠加完整窗口的 20 回合尾随均值。输入文件都通过检查后才写图。
最大购电跳变与 RMS 使用日内相邻 23 个差分，不包含跨日或首尾回环。
--hourly-utility 显式允许仅改变 U(L)=a*L²+b*L 的逐小时系数；核验
curvature=-a、reference=b+2*a*baseline_load，其余配置和外生输入必须相同。

Example:
    .venv/bin/python tools/plot_sac_fixed_day_comparison.py \
        --before runs/sac_fixed_day_500 \
        --after runs/sac_fixed_day_refined_500 \
        --output experiments/sac_fixed_day_refined_500_20261018/figures

依赖 numpy、matplotlib 及本目录 plot_sac_fixed_day.py 的只读输入校验。
默认路径相对脚本所在仓库；显式路径按调用方当前目录解析。
"""

import argparse
import copy
import csv
import json
from pathlib import Path
import platform

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

if __package__:
    from .plot_sac_fixed_day import fingerprint, read_inputs, require, set_style
else:
    from plot_sac_fixed_day import fingerprint, read_inputs, require, set_style


ROOT = Path(__file__).resolve().parents[1]
BEFORE, AFTER, GRAY, GOLD = "#376A9D", "#CD6028", "#959CA5", "#A88A4D"
NOISE_KEYS = ("curvature_std", "demand_noise_std", "participation_noise_std")
MIDDAY_HOURS = tuple(range(11, 16))  # Predeclared h11–15, i.e. 11:00–16:00.


def read_run(path):
    inputs, cfg, env, profile, episodes, profits, values, energy = read_inputs(path)
    complete = json.loads((path / "complete.json").read_text("utf-8"))
    manifest = json.loads((path / "manifest.json").read_text("utf-8"))
    result = json.loads((path / "test_1ep.json").read_text("utf-8"))
    require(cfg["evaluation_every"] == 0 and not list(path.glob("evaluation_*.json")),
            f"{path}: 必须关闭全部内部评价，且无内部评价文件。")
    require(not (path / "failure.json").exists(), f"{path}: 存在训练失败记录。")
    require(complete["steps"] == 12000, f"{path}: 500 回合应为 12,000 个环境步。")
    require(all(cfg["response_parameters"].get(key) == 0 for key in NOISE_KEYS),
            f"{path}: 配置必须显式关闭三种响应噪声。")
    with (path / "training.csv").open(encoding="utf-8", newline="") as stream:
        training = list(csv.DictReader(stream))
    first_update = max(cfg["batch_size"], cfg["warmup_steps"], 1)
    expected_updates = max(0, 12000 - first_update + 1) * cfg.get("updates_per_step", 1)
    require(complete["updates"] == int(training[-1]["updates"]) == expected_updates,
            f"{path}: 更新次数与有效环境步及每步更新次数不一致。")
    require(np.allclose([float(r["learning_return"]) for r in training],
                        profits * cfg["reward_scale"], rtol=1e-11, atol=1e-8),
            f"{path}: 训练奖励不是原利润的固定缩放。")
    require(np.allclose(values["learning_reward"], values["profit"] * cfg["reward_scale"],
                        rtol=1e-11, atol=1e-8), f"{path}: 测试奖励不符合利润恒等式。")
    require(result["seed"] == result["cases"][0]["scenario_seed"] == cfg["reference_seed"],
            f"{path}: 固定日实际种子记录不一致。")
    require(result["cases"][0]["levels"] == dict.fromkeys(("load", "wind", "price"), "deterministic"),
            f"{path}: 测试场景不是 deterministic 档位。")
    require(np.all(values["charge"] * values["discharge"] == 0),
            f"{path}: 同一小时同时充放电。")
    storage, dt = env["storage"], env["params"]["dt"]
    expected_soc = (values["initial_soc"] * storage["eta_self"] +
                    (values["charge"] * storage["eta_ch"] - values["discharge"] / storage["eta_dis"]) * dt)
    require(np.allclose(values["soc"], expected_soc, rtol=0, atol=1e-5),
            f"{path}: SOC 不满足充放电能量平衡。")
    require(abs(energy[0] - storage["soc_init"]) < 1e-5, f"{path}: 初始 SOC 与配置不一致。")
    require(np.isclose(values["wind"].sum() / values["baseline_load"].sum(),
                       cfg["renewable_penetration"], rtol=0, atol=1e-6),
            f"{path}: 固定风电电量份额不一致。")

    # Frozen source is required; archived plots need not carry the model weights.
    for relative, expected in manifest["hashes"].items():
        item = fingerprint(path / "source" / relative)
        require(item["sha256"] == expected, f"{path}: 冻结文件哈希不匹配：{relative}")
        inputs.append(item)
    checkpoint_path = path / "checkpoint_00500.pt"
    checkpoint_record = {
        "path": str(checkpoint_path.resolve()),
        "sha256_reread": False,
        "sha256": None,
        "note": "Checkpoint file is absent; its hash was not reread. Existing provenance records may retain its earlier hash.",
    }
    if checkpoint_path.is_file():
        checkpoint = fingerprint(checkpoint_path)
        inputs.append(checkpoint)
        checkpoint_record.update(sha256_reread=True, sha256=checkpoint["sha256"],
                                 note="Checkpoint bytes hashed for provenance only; model not loaded.")
    inputs.append(fingerprint(path / "effective_generator.json"))
    for extra in ("provenance_test.json", "test_provenance.json", "test_record.json", "archive_manifest.json"):
        if (path / extra).exists():
            inputs.append(fingerprint(path / extra))
    return dict(path=path, inputs=inputs, cfg=cfg, env=env, profile=profile,
                episodes=episodes, profits=profits, values=values, energy=energy,
                checkpoint_record=checkpoint_record)


def utility_record(run):
    """Verify explicit coefficients; label historical coefficients as reconstructed."""
    cfg, env = run["cfg"], run["env"]
    coefficients = cfg.get("hourly_utility", {})
    load = np.asarray(run["profile"]["baseline_load"], dtype=float)
    reference = np.asarray(env["hourly_demand_reference"], dtype=float)
    if coefficients:
        require(isinstance(coefficients, dict) and set(coefficients) == {"a", "b"},
                "hourly_utility 必须只包含 a、b 两组系数。")
        a, b = (np.asarray(coefficients[key], dtype=float) for key in ("a", "b"))
        require(a.shape == b.shape == (24,) and np.isfinite(a).all()
                and np.isfinite(b).all() and np.all(a < 0), "效用系数必须为有限24小时数组，且a<0。")
        require(env.get("hourly_utility") == coefficients, "有效环境中的 a、b 与训练配置不一致。")
        curvature = np.asarray(env["demand"].get("hourly_curvature_mean"), dtype=float)
        require(curvature.shape == (24,) and np.allclose(curvature, -a, rtol=1e-12, atol=1e-12),
                "hourly_curvature_mean 必须等于 -a。")
        require(reference.shape == (24,) and np.allclose(reference, b + 2 * a * load,
                                                        rtol=1e-12, atol=1e-10),
                "hourly_demand_reference 必须由 b + 2*a*原负荷 推导。")
        origin = "explicit hourly_utility coefficients saved in config and effective environment"
    else:
        require(env.get("hourly_utility", {}) == {}, "旧配置无小时效用系数，但有效环境含显式系数。")
        curvature = np.full(24, env["demand"]["curvature_mean"], dtype=float)
        if "hourly_curvature_mean" in env["demand"]:
            require(np.array_equal(np.asarray(env["demand"]["hourly_curvature_mean"]), curvature),
                    "无小时效用配置时不能存在不同的逐小时曲率。")
        a = -curvature
        b = reference - 2 * a * load
        origin = "reconstructed equivalent coefficients from historical constant curvature and saved demand reference; not original saved a/b"
    return {"origin": origin, "a": a.tolist(), "b": b.tolist(),
            "hourly_curvature_mean": curvature.tolist(), "hourly_demand_reference": reference.tolist(),
            "derivation": "U(L)=a*L^2+b*L; curvature=-a; demand_reference=b+2*a*baseline_load"}


def compare_inputs(before, after, hourly_utility=False):
    """Require paired exogenous data; preserve the allowed configuration differences."""
    require(before["profile"] == after["profile"], "前后完整名义 profile 不同，不能作同日比较。")
    for key in ("baseline_load", "grid_price", "wind"):
        require(np.array_equal(before["values"][key], after["values"][key]),
                f"前后实际 {key} 不同，不能作同日比较。")
    configs = [dict(run["cfg"], hourly_utility=run["cfg"].get("hourly_utility", {})) for run in (before, after)]
    if hourly_utility:
        require(bool(configs[1]["hourly_utility"]), "--hourly-utility 要求本次存在显式 a、b 校准。")
        for run in (before, after):
            utility_record(run)
        rest = [{key: value for key, value in cfg.items() if key != "hourly_utility"} for cfg in configs]
        require(rest[0] == rest[1], "本轮只允许 hourly_utility 变化；其他训练、状态、响应、储能配置必须相同。")
        environments = [copy.deepcopy(run["env"]) for run in (before, after)]
        for env in environments:
            env.pop("hourly_utility", None)
            env.pop("hourly_demand_reference")
            env["demand"].pop("hourly_curvature_mean", None)
        require(environments[0] == environments[1], "除小时效用系数及派生量外，前后有效环境必须完全相同。")
    else:
        # Default historical comparison keeps its original strict reference check.
        for key in ("user", "hourly_demand_reference", "params"):
            require(before["env"][key] == after["env"][key], f"前后价格/参考值/结算输入 {key} 不同。")
    return {key: {"before": configs[0].get(key), "after": configs[1].get(key)}
            for key in sorted(set(configs[0]) | set(configs[1]))
            if configs[0].get(key) != configs[1].get(key)}


def response_percent(values):
    require(np.all(values["baseline_load"] > 0), "响应比例要求原始负荷为正。")
    return 100 * (values["baseline_load"] - values["user_load"]) / values["baseline_load"]


def diagnostics(run):
    values = run["values"]
    delta = np.diff(values["grid_purchase"])
    index = int(np.argmax(np.abs(delta)))
    midday = np.asarray(MIDDAY_HOURS)
    response_percent(values)  # Validate the denominator before calculating ratios.
    return {
        "price_h22": float(values["price"][22]), "price_h23": float(values["price"][23]),
        "price_change_h22_to_h23": float(values["price"][23] - values["price"][22]),
        "grid_max_absolute_hourly_change": float(np.abs(delta).max()),
        "grid_max_change_hours": [index, index + 1],
        "grid_max_change_signed": float(delta[index]),
        "grid_rms_hourly_change": float(np.sqrt(np.mean(delta ** 2))),
        "grid_daily_peak": float(values["grid_purchase"].max()),
        "grid_daily_std": float(values["grid_purchase"].std()),
        "daily_profit_raw": float(values["profit"].sum()),
        "midday_reduction_pct": float(100 * (values["baseline_load"][midday] - values["user_load"][midday]).sum()
                                       / values["baseline_load"][midday].sum()),
        "training_updates": max(0, 12000 - max(run["cfg"]["batch_size"], run["cfg"]["warmup_steps"], 1) + 1)
                            * run["cfg"].get("updates_per_step", 1),
    }


def draw(output, before, after, metrics, hourly_utility=False):
    fig, axes = plt.subplots(3, 2, figsize=(17, 14.6))
    fig.subplots_adjust(left=.075, right=.975, top=.825, bottom=.14, hspace=.57, wspace=.22)
    fig.suptitle("同一固定日｜两次 SAC 训练与实际测试对比", x=.075, ha="left", y=.972,
                 fontsize=22, weight="bold")
    fig.text(.075, .934, "每组 500 回合训练 + 1 个 24 小时测试回合  ·  原利润目标保持不变"
             f"  ·  更新次数 {metrics['before']['training_updates']:,} → {metrics['after']['training_updates']:,}",
             fontsize=12, color="#475467")
    fig.legend(handles=[Line2D([], [], color=BEFORE, ls="--", lw=1.8, label="前一次"),
                        Line2D([], [], color=AFTER, lw=2.2, label="本次")],
               loc="upper left", bbox_to_anchor=(.07, .914), ncol=2, frameon=False,
               fontsize=12, handlelength=3, columnspacing=3)

    def setup(ax, title, ylabel, subtitle=""):
        ax.set_title(title, loc="left", pad=35, weight="bold", fontsize=14)
        if subtitle:
            ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.4, color="#586474")
        ax.set(xlabel="小时", ylabel=ylabel, xlim=(0, 24), xticks=np.arange(0, 25, 3))
        ax.grid(axis="y", color="#E1E6ED", linewidth=.7)

    def step(ax, values, color, label=None, ls="-", width=2):
        ax.stairs(values, np.arange(25), baseline=None, color=color, label=label,
                  linestyle=ls, linewidth=width)

    a, b = before["values"], after["values"]
    ax = axes[0, 0]
    setup(ax, "a  训练日利润", "利润（AUD / 日）", "淡线为原始单回合值；主线为过去 20 回合均值")
    for run, color, ls in ((before, BEFORE, "--"), (after, AFTER, "-")):
        values = run["profits"] / 100
        ax.plot(run["episodes"], values, color=color, alpha=.19, lw=.65)
        mean = np.convolve(values, np.ones(20) / 20, mode="valid")
        ax.plot(run["episodes"][19:], mean, color=color, ls=ls, lw=2)
    ax.set(xlabel="训练回合", xlim=(1, 500), xticks=[1, 100, 200, 300, 400, 500])
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    ax = axes[0, 1]
    dm = metrics["before"]["price_change_h22_to_h23"], metrics["after"]["price_change_h22_to_h23"]
    setup(ax, "b  零售价、原 TOU 与批发价", "澳分 / kWh",
          f"h22 → h23 零售价变化：前 {dm[0]:+.3f}  |  本次 {dm[1]:+.3f} 澳分/kWh")
    ax.axvspan(22, 24, color=GOLD, alpha=.07)
    step(ax, before["profile"]["retail_price"], GOLD, "原 TOU", ":", 1.4)
    step(ax, a["grid_price"], GRAY, "批发价", "-", 1.3)
    step(ax, a["price"], BEFORE, ls="--", width=1.8)
    step(ax, b["price"], AFTER, width=2.2)
    ax.axvline(23, color=GOLD, alpha=.45, lw=.8, ls=":")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(.01, .44), ncol=2, fontsize=9)

    ax = axes[1, 0]
    setup(ax, "c  实际储能功率", "功率（kW）", "正值：充电；负值：放电。直接使用实际功率，不用原始动作推断")
    ax.axhline(0, color=GRAY, lw=.8)
    step(ax, a["charge"] - a["discharge"], BEFORE, ls="--", width=1.8)
    step(ax, b["charge"] - b["discharge"], AFTER, width=2.2)

    ax = axes[1, 1]
    setup(ax, "d  SOC（以储能量表示）", "储能量（kWh）", "圆点为每小时边界；点间直线仅辅助阅读")
    for run, color, ls in ((before, BEFORE, "--"), (after, AFTER, "-")):
        ax.plot(np.arange(25), run["energy"], color=color, ls=ls, lw=1.9, marker="o", ms=3)
        for key in ("soc_min", "soc_max"):
            ax.axhline(run["env"]["storage"][key], color=color, lw=.7, alpha=.35, ls=":")
    ax.set_xlabel("小时边界（0 为初始状态）")
    lower = min(before["env"]["storage"]["soc_min"], after["env"]["storage"]["soc_min"])
    upper = max(before["env"]["storage"]["soc_max"], after["env"]["storage"]["soc_max"])
    ax.set_ylim(lower - .08 * (upper - lower), upper + .08 * (upper - lower))

    ax = axes[2, 0]
    subtitle = "固定原负荷相同；前后均为单个测试回合的真实用户用电"
    if hourly_utility:
        old, new = metrics["before"]["midday_reduction_pct"], metrics["after"]["midday_reduction_pct"]
        subtitle = f"中午 h11–15 加权削减：{old:.2f}% → {new:.2f}%（{new - old:+.2f} 个百分点）"
        ax.axvspan(11, 16, color=GOLD, alpha=.10)
    setup(ax, "e  原负荷与响应后负荷", "功率（kW）", subtitle)
    step(ax, a["baseline_load"], GRAY, "原负荷", ":", 1.4)
    step(ax, a["user_load"], BEFORE, ls="--", width=1.8)
    step(ax, b["user_load"], AFTER, width=2.2)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    ax = axes[2, 1]
    x, y = metrics["before"], metrics["after"]
    maximum = f"最大 |Δ购电|：前 {x['grid_max_absolute_hourly_change']:.1f} → 本次 {y['grid_max_absolute_hourly_change']:.1f} kW"
    rms = f"RMS：{x['grid_rms_hourly_change']:.1f} → {y['grid_rms_hourly_change']:.1f} kW"
    setup(ax, "f  电网购电与相邻小时跳变", "功率（kW）", maximum + "；" + rms)
    step(ax, np.maximum(a["baseline_load"] - a["wind"], 0), GRAY, "原负荷扣风电", ":", 1.25)
    step(ax, a["grid_purchase"], BEFORE, ls="--", width=1.8)
    step(ax, b["grid_purchase"], AFTER, width=2.2)
    for values, metric, color in ((a, x, BEFORE), (b, y, AFTER)):
        first, last = metric["grid_max_change_hours"]
        ax.plot([last, last], [values["grid_purchase"][first], values["grid_purchase"][last]],
                color=color, linewidth=4, alpha=.3)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    h_before = " → ".join(f"h{h}" for h in x["grid_max_change_hours"])
    h_after = " → ".join(f"h{h}" for h in y["grid_max_change_hours"])
    interpretation = ("仅 a、b 及派生参考价/曲率变化；原 TOU、其余参数与预算相同。中午比例按原负荷加权；同日测试不证明泛化或收敛。"
                      if hourly_utility else
                      "本图比较整套配置，不能归因于单个改动；相同固定日的训练后测试不证明未见场景泛化或已收敛。")
    footer = (
        f"最大购电跳变发生于：前一次 {h_before}；本次 {h_after}。RMS = sqrt(mean(Δ购电²))，仅计算日内相邻 23 次变化。\n"
        "前后外生数据相同且三项环境噪声均为 0；SAC 训练探索仍有随机性。测试动作、价格、负荷与功率均未平滑。\n"
        + interpretation + "\n"
        "单位沿用本轮假设：澳分/kWh、kW、kWh；训练利润 AUD = 原始利润 ÷ 100。原始数据来源未核定货币单位。"
    )
    fig.text(.075, .026, footer, fontsize=9.2, color="#475467", va="bottom", linespacing=1.55)
    paths = [output / "fixed_day_comparison.png", output / "fixed_day_comparison.pdf"]
    fig.savefig(paths[0], dpi=220)
    fig.savefig(paths[1], metadata={"Title": "固定同一天 SAC：两次500回合训练与单次测试对比",
                                  "CreationDate": None, "ModDate": None})
    plt.close(fig)
    return paths


def write_hourly(path, before, after, hourly_utility=False):
    columns = {"hour": np.arange(24), "tou_price": np.asarray(before["profile"]["retail_price"])}
    for key in ("grid_price", "baseline_load", "wind"):
        columns[key] = before["values"][key]
    columns["baseline_grid"] = np.maximum(columns["baseline_load"] - columns["wind"], 0)
    for label, run in (("before", before), ("after", after)):
        values = run["values"]
        for key, array in values.items():
            if key not in ("hour", "grid_price", "baseline_load", "wind"):
                columns[label + "_" + key] = array
        columns[label + "_signed_storage_power"] = values["charge"] - values["discharge"]
        columns[label + "_price_change_from_previous_hour"] = [None, *np.diff(values["price"])]
        columns[label + "_grid_change_from_previous_hour"] = [None, *np.diff(values["grid_purchase"])]
        columns[label + "_response_pct"] = response_percent(values)
        if hourly_utility:
            record = utility_record(run)
            for key in ("a", "b", "hourly_curvature_mean", "hourly_demand_reference"):
                columns[label + "_utility_" + key] = record[key]
    columns["response_change_pp"] = columns["after_response_pct"] - columns["before_response_pct"]
    columns["midday_window"] = np.isin(np.arange(24), MIDDAY_HOURS).astype(int)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for hour in range(24):
            row = {name: (float(array[hour]) if array[hour] is not None else "") for name, array in columns.items()}
            row["hour"] = hour
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path, default=ROOT / "runs/sac_fixed_day_500")
    parser.add_argument("--after", type=Path, default=ROOT / "runs/sac_fixed_day_refined_500")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/sac_fixed_day_refined_500_20261018/figures")
    parser.add_argument("--font", default="PingFang SC")
    parser.add_argument("--hourly-utility", action="store_true",
                        help="Only hourly a/b calibration may differ; verify its derived curvature/reference")
    args = parser.parse_args()
    require(args.before.resolve() != args.after.resolve(), "前后不能指向同一运行目录。")
    before, after = read_run(args.before.resolve()), read_run(args.after.resolve())
    changes = compare_inputs(before, after, args.hourly_utility)
    metrics = {label: diagnostics(run) for label, run in (("before", before), ("after", after))}
    metrics["delta_after_minus_before"] = {
        (key.removesuffix("_pct") + "_pp" if key.endswith("_pct") else key): metrics["after"][key] - value
        for key, value in metrics["before"].items()
        if isinstance(value, (int, float))
    }
    font_path = set_style(args.font)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = draw(output, before, after, metrics, args.hourly_utility)
    hourly = output / "hourly_comparison.csv"
    write_hourly(hourly, before, after, args.hourly_utility)
    artifacts.append(hourly)
    for run in (before, after):
        require(all(item == fingerprint(item["path"]) for item in run["inputs"]),
                "绘图期间输入发生变化；请等待训练和测试写入完成。")
    manifest = {
        "purpose": "两次固定日实验的只读对比，每次500回合训练、一个已保存24小时测试回合",
        "policy_loaded_or_evaluated": False,
        "training_episodes_each": 500, "test_episodes_each": 1, "test_hours_each": 24,
        "test_smoothing": "none", "training_smoothing": "20-episode trailing arithmetic mean, begins at episode 20",
        "paired_inputs": "Complete nominal profile and realized baseline load, wholesale price, wind are identical.",
        "environment_noise": dict.fromkeys(NOISE_KEYS, 0),
        "grid_jump_definition": "Absolute/signed differences of the 23 adjacent hourly grid-purchase pairs; no midnight wrap. First maximum selected on ties.",
        "grid_rms_definition": "sqrt(mean(diff(grid_purchase)**2)); 23 within-day differences",
        "unit_assumption": {"price": "Australian cents/kWh", "power": "kW", "energy": "kWh",
                            "training_plot_profit": "AUD = raw / 100", "source_units_verified": False},
        "configuration_changes": changes, "metrics": metrics,
        "comparison_mode": "hourly_utility_only" if args.hourly_utility else "historical_strict_reference",
        "hourly_utility": ({label: utility_record(run) for label, run in (("before", before), ("after", after))}
                           if args.hourly_utility else None),
        "response_definition": "100*(baseline_load-user_load)/baseline_load; positive=reduction, negative=increase; changes reported in percentage points",
        "midday_window": {"hours": list(MIDDAY_HOURS), "clock_interval": "11:00–16:00",
                          "selected_before_new_test": True,
                          "aggregation": "100*sum(baseline_load-user_load)/sum(baseline_load), not unweighted mean of hourly ratios"},
        "inputs": {"before": before["inputs"], "after": after["inputs"]},
        "checkpoint_verification": {"before": before["checkpoint_record"], "after": after["checkpoint_record"]},
        "script": fingerprint(__file__), "helper_script": fingerprint(Path(__file__).with_name("plot_sac_fixed_day.py")),
        "outputs": [fingerprint(path) for path in artifacts],
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__, "font": args.font, "font_path": font_path},
        "limitations": "Same fixed-day tests are not unseen data; this comparison cannot isolate individual parameter effects. Existing test files record one saved episode each, not an execution-call audit.",
    }
    destination = output / "comparison_manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps({"outputs": [str(path) for path in artifacts], "manifest": str(destination),
                      "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
