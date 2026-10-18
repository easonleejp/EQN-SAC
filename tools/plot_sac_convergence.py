#!/usr/bin/env python3
"""Plot training and development learning curves for four fixed SAC pilots.

This figure compares the original baseline (seed 42), the experimental market18
configuration (seed 42), and its seed 7/123 repeats. It does not search for the
best run or checkpoint, load a policy, or include final held-out evaluations.
Development points are the saved, deterministic means over the same 27 cases.
The grid peak is the mean of each case's daily peak, not a mean curve's peak.
Currency follows this study's explicit assumption: AUD = recorded profit / 100.
Training curves use daily profit, not scaled learning returns. The trailing mean
first appears once a complete window is available; no padding is introduced.

Run from any working directory (numpy and matplotlib are required)::

    python /path/to/EQN-SAC/tools/plot_sac_convergence.py
    python tools/plot_sac_convergence.py --runs-root runs --output checks/figures

Only convergence.png and convergence.pdf in the output directory are written.
Source data and the archive manifest are never modified. These learning curves
are diagnostic evidence, not a proof of convergence or held-out performance.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT_ROOT / "experiments/sac_parameter_revision_20261018"
METRIC_NAMES = ("profit", "storage_net", "grid_peak")
# Distinct, color-blind-friendly colors; repetitions remain visually secondary.
RUNS = (
    ("baseline", 42, "#2463A2", "-", 2.4, "o"),
    ("forecast18_scale3", 42, "#C65B21", "-", 2.4, "s"),
    ("forecast18_scale3_seed7", 7, "#6A9E8C", "--", 1.25, "^"),
    ("forecast18_scale3_seed123", 123, "#A18AB9", "-.", 1.25, "D"),
)


@dataclass
class Curves:
    name: str
    seed: int
    episodes: np.ndarray
    training_profit: np.ndarray
    evaluation_episodes: np.ndarray
    metrics: dict[str, np.ndarray]
    case_signature: tuple
    evaluation_seed: int


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def case_signature(payload: dict) -> tuple:
    """Require genuinely paired saved cases, including their response RNG seeds."""
    cases = payload["cases"]
    if payload["n"] != 27 or len(cases) != 27 or payload["deterministic"] is not True:
        raise ValueError("Expected a deterministic 27-case development evaluation")
    signature = tuple(
        (case["case"], case["scenario_seed"], case["response_seed"],
         tuple(sorted(case["levels"].items())))
        for case in cases
    )
    if len(set(entry[0] for entry in signature)) != 27:
        raise ValueError("Development evaluation contains duplicate case IDs")
    return signature


def load_curves(root: Path, name: str, expected_seed: int) -> Curves:
    directory = root / f"sac_revision_{name}_2000"
    config = read_json(directory / "config.json")
    complete = read_json(directory / "complete.json")
    if config["seed"] != expected_seed or config["episodes"] != 2000:
        raise ValueError(f"Unexpected configuration in {directory}")
    if complete["episodes"] != config["episodes"]:
        raise ValueError(f"Incomplete training in {directory}")
    with (directory / "training.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    episodes = np.array([int(row["episode"]) for row in records])
    if not np.array_equal(episodes, np.arange(1, config["episodes"] + 1)):
        raise ValueError(f"Noncontiguous or incomplete training log: {directory}")
    profit = np.array([float(row["profit"]) for row in records])
    learning_return = np.array([float(row["learning_return"]) for row in records])
    if not np.all(np.isfinite(profit)) or not np.allclose(
        learning_return, profit * config["reward_scale"], rtol=1e-10, atol=1e-8
    ):
        raise ValueError(f"Invalid profit-only daily rewards: {directory}")

    # Archived intermediate summaries omit hourly traces but retain case metrics.
    files = {int(path.stem.removeprefix("evaluation_")): path
             for path in (directory / "development_summaries").glob("evaluation_*.json")}
    # A live run can be used as well as the reduced archive.
    for path in directory.glob("evaluation_[0-9]*.json"):
        files[int(path.stem.removeprefix("evaluation_"))] = path
    final_alias = directory / "evaluation_final.json"
    if final_alias.exists():
        files[config["episodes"]] = final_alias
    expected = list(range(0, config["episodes"] + 1, config["evaluation_every"]))
    if expected[-1] != config["episodes"]:
        expected.append(config["episodes"])
    if sorted(files) != expected:
        raise ValueError(f"Missing or unexpected development evaluation episodes: {directory}")
    payloads = [read_json(files[episode]) for episode in expected]
    signature = case_signature(payloads[0])
    evaluation_seed = int(payloads[0]["seed"])
    for payload in payloads:
        if case_signature(payload) != signature or payload["seed"] != evaluation_seed:
            raise ValueError(f"Development cases change across checkpoints: {directory}")
        for metric in METRIC_NAMES:
            value = float(payload["metrics"][metric])
            case_values = [case["curves"]["grid"]["peak"] if metric == "grid_peak"
                           else case[metric] for case in payload["cases"]]
            if not np.isfinite(value) or not np.isclose(value, np.mean(case_values),
                                                       rtol=1e-10, atol=1e-8):
                raise ValueError(f"Invalid case mean for {metric}: {directory}")
    metrics = {key: np.array([payload["metrics"][key] for payload in payloads], dtype=float)
               / (1 if key == "grid_peak" else 100) for key in METRIC_NAMES}
    return Curves(name, expected_seed, episodes, profit / 100,
                  np.array(expected), metrics, signature, evaluation_seed)


def select_language(requested: str) -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((name for name in ("PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS")
                         if name in available), None)
    if requested == "zh" and chinese_font is None:
        raise ValueError("No supported Chinese font is installed; use --language en")
    language = "zh" if requested != "en" and chinese_font else "en"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ([chinese_font] if language == "zh" else []) + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#B1BAC4",
        "axes.labelcolor": "#263442",
        "text.color": "#263442",
        "xtick.color": "#536171",
        "ytick.color": "#536171",
        "savefig.facecolor": "white",
    })
    return language


def plot_curves(curves: list[Curves], output: Path, window: int, language: str) -> None:
    zh = language == "zh"
    labels = (["原基线 · seed 42", "market18 实验候选 · seed 42",
               "同一候选 · seed 7", "同一候选 · seed 123"] if zh else
              ["Original baseline · seed 42", "market18 experimental · seed 42",
               "Same candidate · seed 7", "Same candidate · seed 123"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.2))
    fig.subplots_adjust(left=.075, right=.975, bottom=.16, top=.805, hspace=.45, wspace=.23)
    fig.suptitle("纯 SAC 学习曲线与开发场景表现" if zh else
                 "Pure SAC: learning curves and development performance", y=.967,
                 x=.075, ha="left", fontsize=23, fontweight="bold")
    fig.text(.075, .924, "原利润目标保持不变  |  2,000 回合  |  四组训练记录" if zh else
             "Original profit objective unchanged  |  2,000 episodes  |  Four training runs",
             fontsize=12, color="#536171")
    handles = [Line2D([], [], color=style[2], linestyle=style[3], linewidth=style[4],
                      marker=style[5], markersize=5, label=label)
               for style, label in zip(RUNS, labels)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.07, .894), ncol=4,
               frameon=False, fontsize=11, handlelength=2.8, columnspacing=2.2)
    ax_train, ax_profit, ax_storage, ax_peak = axes.flat
    titles = (["A  训练日利润", "B  开发场景平均日利润", "C  开发场景储能边际收益",
               "D  开发场景平均购电峰值"] if zh else
              ["A  Daily training profit", "B  Development mean daily profit",
               "C  Development storage margin", "D  Development mean daily grid peak"])
    subtitles = ([f"淡线：单日原始值；实 / 虚线：过去 {window} 回合均值", "固定 27 场景；确定性策略",
                  "同价格 / 响应路径下，相对零储能的利润差", "每场景先取日最大购电功率，再求平均"] if zh else
                 [f"Faint: raw daily values; strong: trailing {window}-episode mean",
                  "Same 27 cases; deterministic policy",
                  "Profit difference from zero storage at the same prices / response",
                  "Daily maximum grid purchase per case, then mean across cases"])
    for ax, title, subtitle in zip(axes.flat, titles, subtitles):
        ax.set_title(title, loc="left", pad=29, fontweight="bold")
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=10, color="#637080")
        ax.grid(axis="y", color="#E2E7EC", linewidth=.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 2030)
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.set_xlabel("已训练回合" if zh else "Training episodes")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(length=3)
    for ax in (ax_train, ax_profit, ax_storage):
        ax.set_ylabel("AUD / 日" if zh else "AUD / day")
    ax_peak.set_ylabel("kW")

    for curve, style in zip(curves, RUNS):
        _, _, color, linestyle, width, marker = style
        main = curve.seed == 42
        ax_train.plot(curve.episodes, curve.training_profit, color=color, lw=.5,
                      alpha=.13 if main else .045, rasterized=True, zorder=1)
        rolling = np.convolve(curve.training_profit, np.ones(window) / window, mode="valid")
        ax_train.plot(curve.episodes[window - 1:], rolling, color=color, linestyle=linestyle,
                      lw=width, zorder=4 if main else 3)
        for ax, metric in ((ax_profit, "profit"), (ax_storage, "storage_net"), (ax_peak, "grid_peak")):
            ax.plot(curve.evaluation_episodes, curve.metrics[metric], color=color,
                    linestyle=linestyle, lw=width, marker=marker, markersize=4 if main else 3,
                    markeredgecolor="white", markeredgewidth=.5, zorder=4 if main else 3)
    ax_storage.axhline(0, color="#778391", lw=.85, zorder=1)

    # Keep the large initial losses visible while resolving the later sign changes.
    inset = ax_storage.inset_axes([.34, .14, .63, .49])
    inset.set_facecolor("#F7F9FB")
    for curve, style in zip(curves, RUNS):
        keep = curve.evaluation_episodes >= 250
        inset.plot(curve.evaluation_episodes[keep], curve.metrics["storage_net"][keep],
                   color=style[2], linestyle=style[3], lw=style[4] * .75,
                   marker=style[5], markersize=2.5)
    inset.axhline(0, color="#778391", lw=.65)
    inset.set_title("250–2000 回合细节（AUD / 日）" if zh else
                    "Episodes 250–2000 (AUD / day)", fontsize=9, loc="left", pad=5)
    inset.set_xlim(220, 2030)
    inset.set_xticks([500, 1250, 2000])
    inset.tick_params(labelsize=8, length=2)
    inset.grid(axis="y", color="#E2E7EC", lw=.6)

    if zh:
        notes = [
            "口径：训练采用随机场景；开发评价固定 27 个档位组合，seed 从 "
            f"{curves[0].evaluation_seed} 起。开发点之间的连线仅帮助阅读。",
            "货币按本轮单位假设换算（AUD = 原始利润 ÷ 100）；价格响应、原利润目标不变。储能收益和购电峰值仅作诊断。",
            "这些曲线不证明已收敛；开发评价参与了参数筛选，不属于独立最终评价。实验候选尚未推荐替代原基线。",
        ]
    else:
        notes = [
            f"Training uses random scenarios. Development: fixed 27 level combinations, seeds from {curves[0].evaluation_seed}; lines guide the eye.",
            "Study currency assumption: AUD = raw profit / 100. Original response / profit objective unchanged; storage margin and peaks are diagnostics.",
            "These curves do not prove convergence. Development cases informed selection and are not held out. The experimental candidate is not recommended.",
        ]
    for y, note in zip((.105, .076, .047), notes):
        fig.text(.075, y, note, fontsize=10, color="#536171", ha="left")
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = output / f"convergence.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=.2)
        print(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=ARCHIVE / "runs")
    parser.add_argument("--output", type=Path, default=ARCHIVE / "figures")
    parser.add_argument("--window", type=int, default=50, help="Trailing mean window (default: 50 episodes)")
    parser.add_argument("--language", choices=("auto", "zh", "en"), default="auto")
    args = parser.parse_args()
    if not 1 <= args.window <= 2000:
        parser.error("--window must be between 1 and 2000")
    curves = [load_curves(args.runs_root.resolve(), run[0], run[1]) for run in RUNS]
    if any((curve.case_signature, curve.evaluation_seed) !=
           (curves[0].case_signature, curves[0].evaluation_seed) for curve in curves[1:]):
        raise ValueError("Runs do not use the same development cases")
    plot_curves(curves, args.output.resolve(), args.window, select_language(args.language))


if __name__ == "__main__":
    main()
