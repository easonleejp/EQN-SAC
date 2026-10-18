"""Plot saved SAC price, physical dispatch, SOC and load responses.

Reads archived evaluations only. No policies or outcome-dependent scenarios are
selected. Money/power units follow the explicitly documented engineering
assumption; all bands describe scenario variability, not confidence intervals.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GRAY, GREEN = "#2563a6", "#d86a25", "#667085", "#23866f"


def read_run(root, name):
    path = root / name
    evaluation = json.loads((path / "holdout_108.json").read_text())
    environment = json.loads((path / "effective_environment.json").read_text())
    profile = json.loads((path / "effective_profile.json").read_text())
    keys = ("price", "grid_price", "charge", "discharge", "soc", "initial_soc",
            "baseline_load", "user_load", "wind", "grid_purchase")
    arrays = {key: np.asarray([[row[key] for row in case["hourly"]]
                              for case in evaluation["cases"]], dtype=float) for key in keys}
    return evaluation, environment, profile, arrays


def style(font):
    # The task runs on macOS. Fail visibly if the requested CJK font is absent;
    # passing another installed font is supported by the CLI.
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [font],
                         "axes.unicode_minus": False, "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "legend.fontsize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white",
                         "axes.titlepad": 11, "pdf.fonttype": 42})
    font_manager.findfont(font, fallback_to_default=False)


def tidy(ax, title, ylabel, hours=True):
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    if hours:
        ax.set(xlabel="小时", xticks=np.arange(0, 25, 3), xlim=(0, 23))


def mean_line(ax, values, label, color, band=False, x=None, **kwargs):
    x = np.arange(values.shape[1]) if x is None else x
    ax.plot(x, values.mean(axis=0), label=label, color=color, lw=2.2, **kwargs)
    if band:
        lo, hi = np.quantile(values, [.1, .9], axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=.10, linewidth=0)


def save(fig, name, footer, output):
    fig.text(.05, .015, footer, fontsize=9, color="#475467", va="bottom")
    fig.subplots_adjust(left=.08, right=.98, bottom=.12, top=.89, hspace=.48, wspace=.28)
    for extension in ("png", "pdf"):
        fig.savefig(output / (name + "." + extension), dpi=190)
    plt.close(fig)


def actions(base, candidate, profile, output):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4))
    fig.suptitle("纯 SAC｜价格、储能动作与 SOC", x=.05, ha="left", y=.975, fontsize=20, weight="bold")
    fig.text(.05, .925, "原基线与 market18 实验候选 · 均训练 2000 回合，seed 42 · 同一套独立 108 场景", fontsize=11, color="#475467")
    ax = axes[0, 0]
    mean_line(ax, base["price"], "原基线零售价", BLUE)
    mean_line(ax, candidate["price"], "候选零售价", ORANGE)
    ax.plot(np.arange(24), profile["retail_price"], label="原始 TOU", color="#202936", ls="--", lw=1.8)
    mean_line(ax, base["grid_price"], "批发电价", GREEN)
    tidy(ax, "a  零售价与批发价的日内关系", "电价（澳分/kWh，单位假设）")
    ax.legend(ncol=2, loc="lower right", frameon=False)

    ax = axes[0, 1]
    for array, label, color in ((base, "原基线", BLUE), (candidate, "候选", ORANGE)):
        mean_line(ax, array["charge"] - array["discharge"], label, color)
    ax.axhline(0, color=GRAY, lw=.9)
    tidy(ax, "b  实际储能净功率：正充电，负放电", "功率（kW）")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 0]
    for array, label, color in ((base, "原基线", BLUE), (candidate, "候选", ORANGE)):
        energy = np.column_stack((array["initial_soc"][:, 0], array["soc"]))
        mean_line(ax, energy / 1000, label, color, band=True, x=np.arange(25))
    tidy(ax, "c  SOC：实际储存能量", "储能量（MWh）")
    ax.set_xlim(0, 24)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 1]
    for array, label, color in ((base, "原基线", BLUE), (candidate, "候选", ORANGE)):
        mean_line(ax, array["user_load"] - array["baseline_load"], label, color)
    ax.axhline(0, color=GRAY, lw=.9)
    tidy(ax, "d  用户负荷变化：正增加，负削减", "相对原始负荷的变化（kW）")
    ax.legend(frameon=False, ncol=2)
    save(fig, "price_storage_soc", "实线为逐小时场景均值；SOC 阴影为场景 P10–P90，非置信区间。SOC 用能量表示，两方案设备容量不同。\n候选仍出现低价时放电等局部行为，尚未稳定达标；原利润目标未增加跟随或削峰奖惩。", output)


def loads(base, candidate, profile, output):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4))
    fig.suptitle("纯 SAC｜用户响应与电网购电变化", x=.05, ha="left", y=.975, fontsize=20, weight="bold")
    fig.text(.05, .925, "均训练 2000 回合，seed 42 · 108 场景的逐小时均值 · 用户负荷与储能效果分开呈现", fontsize=11, color="#475467")
    ax = axes[0, 0]
    mean_line(ax, base["baseline_load"], "原始用户负荷", GRAY, ls="--")
    mean_line(ax, base["user_load"], "原基线响应后", BLUE)
    mean_line(ax, candidate["user_load"], "候选响应后", ORANGE)
    tidy(ax, "a  用户用电：可见平均填谷与削减", "负荷（kW）")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    unmodified_grid = np.maximum(base["baseline_load"] - base["wind"], 0)
    mean_line(ax, unmodified_grid, "原负荷扣同一风电", GRAY, ls="--")
    mean_line(ax, base["grid_purchase"], "原基线实际购电", BLUE)
    mean_line(ax, candidate["grid_purchase"], "候选实际购电", ORANGE)
    tidy(ax, "b  电网购电：同时包含风电与储能作用", "购电功率（kW）")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 0]
    no_storage = np.maximum(candidate["user_load"] - candidate["wind"], 0)
    demand_effect = no_storage - unmodified_grid
    storage_effect = candidate["grid_purchase"] - no_storage
    mean_line(ax, demand_effect, "需求响应贡献", BLUE)
    mean_line(ax, storage_effect, "储能贡献", ORANGE)
    mean_line(ax, demand_effect + storage_effect, "总购电变化", GREEN)
    ax.axhline(0, color=GRAY, lw=.9)
    tidy(ax, "c  候选购电变化的逐时会计分解", "变化（kW；正增加，负减少）")
    ax.legend(frameon=False, ncol=3, loc="upper right")

    ax = axes[1, 1]
    order = np.argsort(profile["baseline_load"])
    for shift, array, label, color in ((-.19, base, "原基线", BLUE), (.19, candidate, "候选", ORANGE)):
        values = []
        for band in (order[:6], order[-6:]):
            values.append(float((array["user_load"] - array["baseline_load"])[:, band].mean()))
        bars = ax.bar(np.arange(2) + shift, values, width=.34, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:+.0f}", (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 5 if value >= 0 else -7), textcoords="offset points",
                        ha="center", va="bottom" if value >= 0 else "top", fontsize=11)
    ax.axhline(0, color=GRAY, lw=.9)
    tidy(ax, "d  固定低/高负荷时段的用户响应", "用户负荷变化（kW）", hours=False)
    ax.set(xticks=[0, 1], xticklabels=["原负荷最低 6 小时", "原负荷最高 6 小时"], ylim=(-850, 500))
    ax.legend(frameon=False, ncol=2, loc="lower left")
    save(fig, "load_changes", "时段由原始名义负荷确定，未按策略结果重新分类；储能会计对照保持同一价格与用户响应，不是重新优化策略。\n平均曲线的峰值不等于各场景日峰值的均值。原模型无跨时补回，需求削减不能全部解释为负荷转移。", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "experiments/sac_parameter_revision_20261018/runs")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/sac_parameter_revision_20261018/figures")
    parser.add_argument("--font", default="PingFang SC")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style(args.font)
    base_eval, _, profile, base = read_run(args.runs_root, "sac_revision_baseline_2000")
    cand_eval, _, _, candidate = read_run(args.runs_root, "sac_revision_forecast18_scale3_2000")
    if base_eval["seed"] != cand_eval["seed"] or base_eval["n"] != 108 or cand_eval["n"] != 108:
        raise ValueError("Expected paired 108-case holdouts")
    for key in ("baseline_load", "grid_price", "wind"):
        np.testing.assert_array_equal(base[key], candidate[key])
    actions(base, candidate, profile, args.output)
    loads(base, candidate, profile, args.output)
    print(json.dumps({"output": str(args.output), "cases": 108, "seed": base_eval["seed"]}))


if __name__ == "__main__":
    main()
