"""Reproduce the five fixed 500-episode SAC pilot comparisons from saved runs.

This is the 2026-10-18 pilot's analysis protocol, not a generic experiment selector:
baseline, centered, compact, balanced and peak_balanced are required. The candidate
is peak_balanced; its paired differences are compared with baseline and centered.
There are 27 development cases and 108 separate cases (four per each of 27 source
level combinations). We validate the frozen generator's actual scenario curves
before applying a paired, within-stratum bootstrap with 2,000 resamples.

Only comparison.csv, comparison.json, development_comparison.png and
heldout_dispatch.png are written. Frozen source, case results and checkpoints are
never copied or changed. Statistics are conditional on one trained seed; these
five pilot configurations do not establish convergence or multi-seed superiority.
"""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .analyze_dispatch import analyze_evaluation
else:
    from analyze_dispatch import analyze_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("baseline", "centered", "compact", "balanced", "peak_balanced")
LABELS = dict(baseline="SAC baseline", centered="Centered profit", compact="Compact observation",
              balanced="Load-shape objective", peak_balanced="Peak + load shape")
COLORS = dict(baseline="#7a8490", centered="#14866d", compact="#9666b4",
              balanced="#357bc1", peak_balanced="#d36a25")
LEVEL_COMBINATIONS = tuple(itertools.product(("low", "medium", "high"), repeat=3))
EXPECTED_VARIANTS = {
    "baseline": ("full19", "profit", 0.0, 0.0),
    "centered": ("full19", "incremental_profit", 0.0, 0.0),
    "compact": ("causal11", "incremental_profit", 0.0, 0.0),
    "balanced": ("full19", "incremental_profit", 0.5, 0.0),
    "peak_balanced": ("full19", "incremental_profit", 0.5, 1.0),
}
# Fixed study protocol, not the wall-clock execution date or an inferred RNG seed.
BOOTSTRAP_SEED = 20261018
BOOTSTRAP_RESAMPLES = 2000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_metric(case, key):
    if key in case:
        return float(case[key])
    for curve, metrics in case["curves"].items():
        for metric, value in metrics.items():
            if key == curve + "_" + metric:
                return float(value)
    raise ValueError(f"Unknown saved metric: {key}")


def verify_evaluation(payload, count, label):
    """Check case identities, horizon, finite daily metrics and saved aggregation."""
    require(payload["n"] == count and len(payload["cases"]) == count,
            f"{label}: expected exactly {count} cases")
    require(payload.get("deterministic") is True, f"{label}: expected deterministic policy evaluation")
    for i, case in enumerate(payload["cases"]):
        require(case["case"] == i, f"{label}: cases must be ordered 0..{count - 1}")
        require(case["scenario_seed"] == payload["seed"] + i, f"{label}: scenario seed/order mismatch")
        require(case["response_seed"] == case["scenario_seed"] + 700000,
                f"{label}: expected this pilot's per-case response seed protocol")
        for key in ("hourly", "reference_hourly"):
            require(len(case[key]) == 24 and [r["hour"] for r in case[key]] == list(range(24)),
                    f"{label}: {key} must contain all 24 ordered hours")
    for key, saved in payload["metrics"].items():
        values = np.array([case_metric(c, key) for c in payload["cases"]])
        require(np.isfinite(values).all() and np.isfinite(saved), f"{label}: nonfinite {key}")
        require(np.isclose(values.mean(), saved, rtol=1e-12, atol=1e-8),
                f"{label}: saved {key} does not equal the mean of its cases")


def verify_paired(reference, candidate, label):
    """Strict lengths are checked first, so zip cannot silently truncate cases."""
    require(reference["n"] == candidate["n"] and len(reference["cases"]) == len(candidate["cases"]),
            f"{label}: unequal paired case counts")
    for a, b in zip(reference["cases"], candidate["cases"]):
        require((a["case"], a["scenario_seed"], a["response_seed"]) ==
                (b["case"], b["scenario_seed"], b["response_seed"]), f"{label}: case identity mismatch")
        require(a["reference_hourly"] == b["reference_hourly"], f"{label}: reference policy differs")
        require(len(a["hourly"]) == len(b["hourly"]) == 24, f"{label}: incomplete paired hours")
        for x, y in zip(a["hourly"], b["hourly"]):
            require(all(x[k] == y[k] for k in ("baseline_load", "wind", "grid_price")),
                    f"{label}: paired exogenous inputs differ")


def frozen_generator(run):
    """Load the run's hash-verified generator without modifying sys.path or cwd."""
    path = run["path"] / "source/eqnsac/scenarios.py"
    module_name = "_sac_pilot_frozen_scenarios_" + sha256(path)[:12]
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    require(module_spec is not None and module_spec.loader is not None, "Cannot load frozen scenario generator")
    module = importlib.util.module_from_spec(module_spec)
    # dataclasses resolves annotations through sys.modules during class creation.
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    profile = {k: np.asarray(v, dtype=np.float32) for k, v in
               read_json(run["path"] / "source/data/profile.json").items()}
    return module.ThreeSourceGenerator(profile)


def verify_strata(payload, generator, expected_replicates, label):
    """Reconstruct inputs to verify the index-based strata missing from older logs."""
    groups = [[] for _ in LEVEL_COMBINATIONS]
    for case in payload["cases"]:
        group = case["case"] % len(LEVEL_COMBINATIONS)
        scenario = generator.sample(case["scenario_seed"], *LEVEL_COMBINATIONS[group])
        for recorded, generated in (("baseline_load", "baseline_load"), ("wind", "wind_power"), ("grid_price", "grid_price")):
            expected = np.asarray(scenario[generated], dtype=np.float32).astype(float)
            actual = np.array([row[recorded] for row in case["hourly"]])
            require(np.array_equal(expected, actual), f"{label}: case {case['case']} does not match its {LEVEL_COMBINATIONS[group]} stratum")
        groups[group].append(case["case"])
    require(all(len(g) == expected_replicates for g in groups), f"{label}: incomplete or unequal strata")
    return groups


def load_runs(runs_root):
    runs = {}
    for name in METHODS:
        path = runs_root / f"sac_{name}_500"
        cfg, manifest, complete = [read_json(path / file) for file in ("config.json", "manifest.json", "complete.json")]
        variant = (cfg["observation"], cfg["reward"], cfg["shape_weight"], cfg.get("peak_weight", 0.0))
        require(variant == EXPECTED_VARIANTS[name], f"{name}: configuration differs from the fixed pilot protocol")
        require(cfg["scenario_mode"] == "three_source" and cfg["response_rng"] == "per_case",
                f"{name}: this comparison requires three-source/per-case sampling")
        require(cfg["episodes"] == complete["episodes"] == 500 and complete["steps"] == 12000,
                f"{name}: incomplete 500-episode/12,000-step run")
        require(not (path / "failure.json").exists(), f"{name}: failure marker present")
        for relative, digest in manifest["hashes"].items():
            frozen = path / "source" / relative
            require(frozen.is_file() and sha256(frozen) == digest, f"{name}: frozen source hash mismatch: {relative}")
        require((path / "checkpoint_00500.pt").is_file(), f"{name}: final checkpoint missing")
        with (path / "training.csv").open(encoding="utf-8-sig", newline="") as stream:
            training = list(csv.DictReader(stream))
        require([int(r["episode"]) for r in training] == list(range(1, 501)), f"{name}: incomplete training.csv")
        require(int(training[-1]["updates"]) == complete["updates"], f"{name}: final update count mismatch")
        development = {}
        for file in sorted(path.glob("evaluation_*.json")):
            episode = int(file.stem.removeprefix("evaluation_"))
            require(episode not in development, f"{name}: duplicate development episode")
            value = read_json(file)
            verify_evaluation(value, 27, f"{name}/{file.name}")
            development[episode] = value
        require(sorted(development) == list(range(0, 501, 50)), f"{name}: expected validation at episodes 0,50,...,500")
        holdout = read_json(path / "holdout_108.json")
        verify_evaluation(holdout, 108, name + "/holdout_108.json")
        spec = read_json(path / "source/data/environment.json")
        analysis = analyze_evaluation(holdout, spec)
        require(analysis["checks_passed"], f"{name}: hourly physical checks failed")
        for saved, checked in zip(holdout["cases"], analysis["cases"]):
            recomputed = {
                "profit": checked["economics"]["profit"],
                "storage_net": checked["economics"]["storage_marginal_profit_reward"],
                **{"grid_" + key: checked["curves"]["grid"][key]
                   for key in ("peak", "peak_valley", "std")},
            }
            require(all(np.isclose(case_metric(saved, key), value, rtol=1e-12, atol=1e-8)
                        for key, value in recomputed.items()),
                    f"{name}: daily comparison metrics disagree with their hourly trace")
        runs[name] = dict(path=path, config=cfg, manifest=manifest, complete=complete,
                          development=development, holdout=holdout, spec=spec)
    reference = runs["baseline"]
    varying = {"observation", "reward", "shape_weight", "peak_weight"}
    common_config = {k: v for k, v in reference["config"].items() if k not in varying}
    generator = frozen_generator(reference)
    strata = verify_strata(reference["holdout"], generator, 4, "baseline holdout")
    development_reference = reference["development"][0]
    verify_strata(development_reference, generator, 1, "baseline development")
    development_seeds = {c["scenario_seed"] for c in development_reference["cases"]}
    holdout_seeds = {c["scenario_seed"] for c in reference["holdout"]["cases"]}
    require(development_seeds.isdisjoint(holdout_seeds), "Development and holdout scenario seeds overlap")
    for name, run in runs.items():
        require({k: v for k, v in run["config"].items() if k not in varying} == common_config,
                f"{name}: shared training settings differ")
        for source in ("eqnsac/scenarios.py", "eqnsac/environment.py", "data/profile.json", "data/environment.json"):
            require(run["manifest"]["hashes"][source] == reference["manifest"]["hashes"][source],
                    f"{name}: scenario or environment source differs")
        verify_paired(reference["holdout"], run["holdout"], name + " holdout")
        for episode, value in run["development"].items():
            verify_paired(development_reference, value, f"{name} development/{episode}")
    return runs, strata


def paired_comparisons(runs, strata):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Draw once, then use the SAME paired case indices for every metric/reference.
    indices = np.stack([np.concatenate([np.array(g)[rng.integers(len(g), size=len(g))]
                                        for g in strata]) for _ in range(BOOTSTRAP_RESAMPLES)])
    comparisons = {}
    for reference in ("baseline", "centered"):
        per_metric = {}
        for key in ("profit", "storage_net", "grid_peak", "grid_peak_valley", "grid_std"):
            candidate = np.array([case_metric(c, key) for c in runs["peak_balanced"]["holdout"]["cases"]])
            base = np.array([case_metric(c, key) for c in runs[reference]["holdout"]["cases"]])
            delta = candidate - base
            reference_mean = float(base.mean())
            per_metric[key] = dict(candidate_mean=float(candidate.mean()), reference_mean=reference_mean,
                                   delta=float(delta.mean()),
                                   percent=float((candidate.mean() / reference_mean - 1) * 100) if reference_mean > 0 else None,
                                   paired_stratified_case_bootstrap_ci95=np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist())
        comparisons[reference] = per_metric
    return comparisons


def receipt(runs, strata, comparisons):
    baseline = runs["baseline"]
    return dict(
        protocol="2026-10-18 five-configuration, single-training-seed pilot",
        seed=baseline["config"]["seed"], episodes_per_method=baseline["complete"]["episodes"],
        methods=list(METHODS), training_steps_per_method=baseline["complete"]["steps"],
        validation=dict(cases=baseline["development"][0]["n"], seed=baseline["development"][0]["seed"],
                        use="development; inspected repeatedly"),
        held_out=dict(cases=baseline["holdout"]["n"], seed=baseline["holdout"]["seed"],
                      weights="27 level combinations equally weighted, four realizations each",
                      use="Study protocol: evaluated after selecting the five pilot configurations; saved files alone cannot establish tuning history."),
        bootstrap=dict(seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, interval="percentile 2.5% and 97.5% of paired mean differences",
                       strata=[dict(levels=list(levels), case_indices=group) for levels, group in zip(LEVEL_COMBINATIONS, strata)]),
        percent_definition="100 * (candidate_mean / reference_mean - 1), only when reference_mean > 0; otherwise null. Delta remains candidate minus reference.",
        caution="One training seed; intervals describe scenario variation conditional on these trained models, not variability across retraining seeds or real-world forecasts.",
        paired_inputs_identical=True, strata_verified_from_frozen_generator=True,
        frozen_source_hashes_verified=True, comparisons=comparisons,
    )


def plot_comparisons(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for name in METHODS:
        development = runs[name]["development"]
        for ax, key, title in zip(axes.flat, ("profit", "grid_peak", "grid_peak_valley", "grid_std"),
                ("Mean daily profit (source units)", "Mean daily grid peak", "Mean daily peak-to-valley gap", "Mean within-day grid standard deviation")):
            ax.plot(list(development), [v["metrics"][key] for v in development.values()],
                    label=LABELS[name], color=COLORS[name], linewidth=1.8)
            ax.set(title=title, xlabel="Training episodes")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Pure SAC: five controlled 500-episode pilot runs\nFixed 27-case development validation; curves do not establish convergence", fontsize=13)
    fig.savefig(output / "development_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)
    hours = np.arange(24)
    baseline_cases = runs["baseline"]["holdout"]["cases"]
    reference = np.mean([[max(r["user_load"] + r["charge"] - r["discharge"] - c["hourly"][i]["wind"], 0)
                          for i, r in enumerate(c["reference_hourly"])] for c in baseline_cases], axis=0)
    axes[0].plot(hours, reference, "--", color="#333333", label="Nominal tariff, no storage")
    for name in ("baseline", "centered", "peak_balanced"):
        cases = runs[name]["holdout"]["cases"]
        for ax, key in zip(axes, ("grid_purchase", "price", "battery_power", "soc")):
            # Charge/discharge are actual powers: works even when P_ch_max != P_dis_max.
            values = np.mean([[r["charge"] - r["discharge"] if key == "battery_power" else r[key]
                               for r in c["hourly"]] for c in cases], axis=0)
            ax.plot(hours, values, color=COLORS[name], label=LABELS[name], linewidth=2)
    axes[2].axhline(0, color="gray", linewidth=.8)
    user = runs["baseline"]["spec"]["user"]
    for bound in ("price_min", "price_max"):
        axes[1].axhline(float(np.mean(user[bound])), color="gray", linestyle=":", linewidth=.8)
    for ax, title, ylabel in zip(axes,
            ("Grid imports", "Retail price", "Executed battery power: charge + / discharge -", "Stored energy"),
            ("Power", "Price", "Power", "Energy")):
        ax.set(title=title, xlabel="Hour", ylabel=ylabel + " (source units)", xticks=np.arange(0, 24, 2))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Separate 108-case evaluation: mean hourly dispatch\nReported peaks/gaps are averaged per day, not calculated from this mean curve", fontsize=13)
    fig.savefig(output / "heldout_dispatch.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments/sac_500ep_20261018")
    args = parser.parse_args()
    runs, strata = load_runs(args.runs_root.resolve())
    comparisons = paired_comparisons(runs, strata)
    report = receipt(runs, strata, comparisons)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [dict(method=name, **runs[name]["holdout"]["metrics"]) for name in METHODS]
    with (args.output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_comparisons(runs, args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), methods=list(METHODS),
                          paired_cases=108, strata=27, resamples=BOOTSTRAP_RESAMPLES,
                          physical_and_pairing_checks="passed"), ensure_ascii=False))


if __name__ == "__main__":
    main()
