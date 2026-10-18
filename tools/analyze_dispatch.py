"""Analyze saved hourly dispatch and learning curves without loading an agent.

Examples:
  python tools/analyze_dispatch.py --evaluation experiments/sac_causal13_500_20261018/run/test_1ep.json --environment experiments/sac_causal13_500_20261018/run/effective_environment.json --output checks/dispatch
  python tools/analyze_dispatch.py --evaluation evaluation_00500.json \
      --environment data/environment.json --output checks/dispatch

Input is the SAC evaluator's JSON: cases[].hourly and optional reference_hourly.
Each hourly row needs hour, baseline_load, wind, user_load, charge, discharge,
soc and price; grid_purchase and initial_soc are checked when present. Reference
rows may omit exogenous inputs because the evaluator reuses the same scenario.
The environment specification supplies dt, efficiencies, SOC and price bounds.
Use rollout_environment() to obtain this format from any fixed Python policy.
No EQN/SAC policy modules or checkpoint loaders are imported by this tool.
"""

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np


DEFINITIONS = {
    "units": "Power, energy, price and profit retain the source environment units; energy = power * dt_hours.",
    "baseline_grid": "max(baseline_load - wind, 0): arithmetic no-response/no-storage benchmark, not a simulated tariff policy.",
    "reference_grid": "The saved paired reference policy: nominal hourly tariff and no storage in the SAC evaluator; demand response is still active.",
    "response_grid_without_storage": "max(actual user_load - wind, 0): hold the actual response fixed and remove storage arithmetically.",
    "grid": "Nonnegative grid imports; surplus is discarded, not exported. Surplus may include battery discharge.",
    "par": "Peak / arithmetic mean; null when mean is zero.",
    "std": "Population standard deviation across hourly powers (ddof=0).",
    "ramps": "Differences between adjacent recorded powers; no last-to-first wrap. Ramp rates divide by dt_hours.",
    "price_saturation": "Within 1% of the allowed mean retail price range by default; matches raw price action >= 0.98 / <= -0.98.",
    "storage_cycles": "Cell-side charge plus discharge throughput divided by twice usable SOC capacity; partial cycles included.",
    "paired_reductions": "Reference minus policy for peak/gap/std/PAR; policy minus reference for valley lift. Positive indicates the stated improvement.",
    "training_curves": "Trailing-window arithmetic means of sampled training episodes; stochastic scenarios/actions differ. Fixed paired evaluation is the policy-comparison evidence.",
    "evaluation_split": "Periodic evaluation files from a training run are development validation, not untouched holdout evidence.",
}


def _number(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def curve_metrics(values, dt_hours=1.0):
    """Summarize a nonnegative power curve, allowing short hand-worked examples."""
    a = np.asarray(values, dtype=float)
    dt = _number(dt_hours, "dt_hours")
    if a.ndim != 1 or not a.size or not np.isfinite(a).all() or np.any(a < 0) or dt <= 0:
        raise ValueError("Expected a nonempty finite nonnegative curve and positive dt_hours")
    mean = float(a.mean())
    ramps = np.diff(a) / dt
    return dict(peak=float(a.max()), valley=float(a.min()), peak_valley=float(np.ptp(a)),
                mean=mean, par=float(a.max() / mean) if mean else None,
                std=float(a.std()), cv=float(a.std() / mean) if mean else None,
                energy=float(a.sum() * dt),
                mean_abs_ramp=float(np.abs(ramps).mean()) if ramps.size else 0.0,
                rms_ramp=float(np.sqrt(np.mean(ramps**2))) if ramps.size else 0.0,
                max_abs_ramp=float(np.abs(ramps).max()) if ramps.size else 0.0)


def _columns(rows, require_exogenous=True):
    if not isinstance(rows, list) or not rows:
        raise ValueError("hourly must be a nonempty list")
    names = ["hour", "user_load", "charge", "discharge", "soc", "price"]
    if require_exogenous:
        names += ["baseline_load", "wind"]
    result = {}
    for name in names:
        try:
            result[name] = np.array([_number(r[name], name) for r in rows])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Every hourly row requires {name}") from exc
    hours = result["hour"]
    if not np.array_equal(hours, np.arange(len(rows))):
        raise ValueError("Hourly rows must be ordered consecutively from hour 0")
    for name in ("user_load", "charge", "discharge", "baseline_load", "wind"):
        if name in result and np.any(result[name] < 0):
            raise ValueError(f"{name} must be nonnegative")
    return result


def _optional(rows, name):
    present = [name in row for row in rows]
    if any(present) and not all(present):
        raise ValueError(f"Optional field {name} must be present in every row or none")
    return np.array([_number(r[name], name) for r in rows]) if all(present) else None


def _policy(rows, columns, wind, spec, saturation_margin, atol, rtol):
    s = spec["storage"]
    dt = _number(spec["params"]["dt"], "dt")
    eta_ch, eta_dis, eta_self = [_number(s[k], k) for k in ("eta_ch", "eta_dis", "eta_self")]
    low_soc, high_soc = [_number(s[k], k) for k in ("soc_min", "soc_max")]
    if dt <= 0 or not 0 < min(eta_ch, eta_dis) <= max(eta_ch, eta_dis) <= 1 or not 0 <= eta_self <= 1 or high_soc <= low_soc:
        raise ValueError("Invalid storage efficiencies, SOC bounds or timestep")
    initial = _optional(rows, "initial_soc")
    starting_soc = float(initial[0]) if initial is not None else _number(s["soc_init"], "soc_init")
    previous = np.r_[starting_soc, columns["soc"][:-1]]
    charge, discharge = columns["charge"], columns["discharge"]
    expected_soc = previous * eta_self + (eta_ch * charge - discharge / eta_dis) * dt
    net = columns["user_load"] + charge - discharge - wind
    expected_grid = np.maximum(net, 0)
    grid = _optional(rows, "grid_purchase")
    grid = expected_grid if grid is None else grid
    if np.any(grid < 0):
        raise ValueError("grid_purchase must be nonnegative")
    surplus = np.maximum(-net, 0)
    checks = {
        "soc_balance_ok": bool(np.allclose(columns["soc"], expected_soc, atol=atol, rtol=rtol)),
        "soc_balance_max_abs_error": float(np.abs(columns["soc"] - expected_soc).max()),
        "grid_balance_ok": bool(np.allclose(grid, expected_grid, atol=atol, rtol=rtol)),
        "grid_balance_max_abs_error": float(np.abs(grid - expected_grid).max()),
        "initial_soc_chain_ok": initial is None or bool(np.allclose(initial, previous, atol=atol, rtol=rtol)),
        "soc_bounds_ok": bool(np.all((np.r_[starting_soc, columns["soc"]] >= low_soc - atol) & (np.r_[starting_soc, columns["soc"]] <= high_soc + atol))),
        "simultaneous_charge_discharge_hours": int(np.sum((charge > atol) & (discharge > atol))),
    }
    if "p_ch_max" in s and "p_dis_max" in s:
        checks["power_bounds_ok"] = bool(np.all(charge <= s["p_ch_max"] + atol) and np.all(discharge <= s["p_dis_max"] + atol))
    response = _optional(rows, "total_response")
    if response is not None and "baseline_load" in columns:
        error = columns["user_load"] + response - columns["baseline_load"]
        checks["response_balance_ok"] = bool(np.allclose(columns["user_load"] + response, columns["baseline_load"], atol=atol, rtol=rtol))
        checks["response_balance_max_abs_error"] = float(np.abs(error).max())
    price_low = float(np.mean(spec["user"]["price_min"]))
    price_high = float(np.mean(spec["user"]["price_max"]))
    if not np.isfinite([price_low, price_high]).all() or price_high <= price_low:
        raise ValueError("Invalid price bounds")
    p = columns["price"]
    margin = saturation_margin * (price_high - price_low)
    prices = dict(mean=float(p.mean()), minimum=float(p.min()), maximum=float(p.max()), std=float(p.std()),
                  lower_fraction=float(np.mean(p <= price_low + margin)),
                  upper_fraction=float(np.mean(p >= price_high - margin)),
                  out_of_bounds_hours=int(np.sum((p < price_low - atol) | (p > price_high + atol))))
    cell_charge = float(charge.sum() * eta_ch * dt)
    cell_discharge = float(discharge.sum() / eta_dis * dt)
    storage = dict(initial_soc=starting_soc, final_soc=float(columns["soc"][-1]),
                   soc_change=float(columns["soc"][-1] - starting_soc),
                   charge_energy=float(charge.sum() * dt), discharge_energy=float(discharge.sum() * dt),
                   bus_throughput=float((charge + discharge).sum() * dt),
                   cell_throughput=cell_charge + cell_discharge,
                   equivalent_full_cycles=(cell_charge + cell_discharge) / (2 * (high_soc - low_soc)),
                   conversion_losses=float(((1 - eta_ch) * charge + (1 / eta_dis - 1) * discharge).sum() * dt),
                   self_discharge_losses=float((previous * (1 - eta_self)).sum()),
                   surplus_energy=float(surplus.sum() * dt))
    economics = {}
    for name in ("profit", "storage_marginal_profit_reward", "learning_reward", "shape_cost"):
        values = _optional(rows, name)
        if values is not None:
            economics[name] = float(values.sum())
    checks["all_ok"] = all(value for name, value in checks.items() if name.endswith("_ok")) and checks["simultaneous_charge_discharge_hours"] == 0 and prices["out_of_bounds_hours"] == 0
    return dict(curves=dict(grid=curve_metrics(grid, dt), user_load=curve_metrics(columns["user_load"], dt)),
                storage=storage, prices=prices, economics=economics, checks=checks)


def analyze_case(case, spec, saturation_margin=0.01, atol=1e-3, rtol=1e-7):
    """Analyze one evaluator case; report numerical inconsistencies, reject malformed data."""
    if not 0 <= saturation_margin < 0.5 or atol < 0 or rtol < 0:
        raise ValueError("Invalid saturation margin or numerical tolerance")
    rows = case["hourly"]
    c = _columns(rows)
    result = _policy(rows, c, c["wind"], spec, saturation_margin, atol, rtol)
    dt = spec["params"]["dt"]
    result["curves"].update(
        baseline_load=curve_metrics(c["baseline_load"], dt),
        baseline_grid=curve_metrics(np.maximum(c["baseline_load"] - c["wind"], 0), dt),
        response_grid_without_storage=curve_metrics(np.maximum(c["user_load"] - c["wind"], 0), dt),
    )
    result["metadata"] = {k: case[k] for k in ("case", "scenario_seed", "response_seed") if k in case}
    reference_rows = case.get("reference_hourly")
    if reference_rows is not None:
        ref = _columns(reference_rows, require_exogenous=False)
        if len(reference_rows) != len(rows):
            raise ValueError("Reference and policy horizons differ")
        for key in ("baseline_load", "wind", "grid_price"):
            a, b = _optional(rows, key), _optional(reference_rows, key)
            if a is not None and b is not None and not np.allclose(a, b, atol=atol, rtol=rtol):
                raise ValueError(f"Reference and policy {key} differ")
        ref["baseline_load"] = c["baseline_load"]
        reference = _policy(reference_rows, ref, c["wind"], spec, saturation_margin, atol, rtol)
        if not np.isclose(result["storage"]["initial_soc"], reference["storage"]["initial_soc"], atol=atol, rtol=rtol):
            raise ValueError("Reference and policy initial SOC differ")
        result["reference"] = reference
        actual, base = result["curves"]["grid"], reference["curves"]["grid"]
        comparison = {}
        for metric in ("peak", "peak_valley", "std", "par", "rms_ramp"):
            a, b = actual[metric], base[metric]
            comparison[metric + "_reduction"] = b - a if a is not None and b is not None else None
            comparison[metric + "_reduction_fraction"] = (b - a) / b if a is not None and b is not None and b != 0 else None
        comparison["valley_lift"] = actual["valley"] - base["valley"]
        comparison["grid_energy_change"] = actual["energy"] - base["energy"]
        if "profit" in result["economics"] and "profit" in reference["economics"]:
            comparison["profit_change"] = result["economics"]["profit"] - reference["economics"]["profit"]
        result["paired_comparison"] = comparison
    return result


def rollout_environment(env, case, policy, response_seed=42):
    """Evaluate policy(state)->physical environment action, with no policy-library dependency.

    These are Environment action coordinates, not SAC's feasible-affine raw storage
    coordinates. A zero storage action means no charging/discharging here.
    """
    state = env.reset(case, response_seed=response_seed)
    rows = []
    done = False
    while not done:
        hour, starting_soc = env.t, env.soc
        state, _, done, info = env.step(policy(state.copy()))
        row = dict(info, initial_soc=float(starting_soc), baseline_load=float(env.load[hour]),
                   wind=float(env.wind[hour]), grid_price=float(env.price[hour]))
        row["grid_purchase"] = max(row["user_load"] + row["charge"] - row["discharge"] - row["wind"], 0)
        rows.append(row)
    return dict(response_seed=response_seed, hourly=rows)


def _flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(_flatten(item, name + "."))
        elif item is None or isinstance(item, (int, float, str, bool)):
            result[name] = item
    return result


def analyze_evaluation(payload, spec, **kwargs):
    cases = [analyze_case(case, spec, **kwargs) for case in payload["cases"]]
    if not cases:
        raise ValueError("Evaluation must contain cases")
    flattened = [_flatten(case) for case in cases]
    means = {}
    for key in flattened[0]:
        if key.startswith(("metadata.", "checks.", "reference.checks.")):
            continue
        values = [row.get(key) for row in flattened]
        if all(v is not None and isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            means[key] = dict(mean=float(np.mean(values)), std=float(np.std(values)), n=len(values))
    return dict(n=len(cases), checks_passed=all(c["checks"]["all_ok"] and c.get("reference", {}).get("checks", {}).get("all_ok", True) for c in cases),
                aggregate=means, cases=cases)


def analyze_learning(rows, window=20):
    """Return numeric training rows plus trailing means; blanks remain missing."""
    if not isinstance(window, int) or window < 1 or not rows:
        raise ValueError("Expected nonempty training rows and positive integer window")
    numeric = [{key: None if value in (None, "") else _number(value, key) for key, value in row.items()} for row in rows]
    episodes = [row.get("episode") for row in numeric]
    if any(e is None or e != int(e) for e in episodes) or any(b <= a for a, b in zip(episodes, episodes[1:])):
        raise ValueError("Training episodes must be strictly increasing integers")
    for row in numeric:
        if row.get("profit") is not None and row.get("reference_profit") is not None:
            row["profit_change"] = row["profit"] - row["reference_profit"]
    names = [key for key in numeric[0] if key not in ("episode", "seconds", "updates")]
    smoothed = []
    for i, row in enumerate(numeric):
        entry = dict(row)
        for name in names:
            values = [r.get(name) for r in numeric[max(0, i + 1 - window):i + 1] if r.get(name) is not None]
            entry["rolling_" + name] = float(np.mean(values)) if values else None
        smoothed.append(entry)
    summary = {}
    for name in names:
        first = [r.get(name) for r in numeric[:window] if r.get(name) is not None]
        last = [(r["episode"], r[name]) for r in numeric[-window:] if r.get(name) is not None]
        summary[name] = dict(first_window_mean=float(np.mean(first)) if first else None,
                             last_window_mean=float(np.mean([v for _, v in last])) if last else None,
                             last_window_slope_per_episode=float(np.polyfit(*np.array(last).T, 1)[0]) if len(last) >= 2 else None,
                             available_count=sum(r.get(name) is not None for r in numeric))
    return dict(episodes=len(rows), first_episode=int(episodes[0]), last_episode=int(episodes[-1]),
                window=window, summary=summary, rows=smoothed)


def _write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plots(out, evaluations, analyses, learning, spec):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "Matplotlib unavailable; CSV and JSON were written."
    if learning:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        rows = learning["rows"]
        for ax, metric in zip(axes.flat, ("profit_change", "grid_peak", "critic_loss", "alpha")):
            y = [r.get(metric) for r in rows]
            ax.plot([r["episode"] for r in rows], y, alpha=.25, linewidth=.6)
            ax.plot([r["episode"] for r in rows], [r.get("rolling_" + metric) for r in rows])
            ax.set(title=metric, xlabel="Training episode")
            ax.grid(alpha=.2)
        fig.savefig(out / "learning_curves.png", dpi=160)
        plt.close(fig)
    if all(episode is not None for episode, _ in evaluations):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for ax, key, label in zip(axes.flat,
                ("paired_comparison.profit_change", "paired_comparison.peak_reduction", "curves.grid.par", "storage.bus_throughput"),
                ("Profit improvement vs paired reference", "Grid peak reduction vs paired reference", "Grid peak-to-average ratio", "Battery throughput (energy)")):
            y = [r["analysis"]["aggregate"].get(key, {}).get("mean") for r in analyses]
            ax.plot([episode for episode, _ in evaluations], y, marker="o", markersize=3)
            ax.set(title=label, xlabel="Training episode")
            ax.grid(alpha=.2)
        fig.suptitle("Fixed development validation cases; means of per-day metrics")
        fig.savefig(out / "validation_curves.png", dpi=160)
        plt.close(fig)
    payload = evaluations[-1][1]
    cases = payload["cases"]
    if len({len(c["hourly"]) for c in cases}) == 1:
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), constrained_layout=True)
        def average(key):
            return np.mean([[row[key] for row in case["hourly"]] for case in cases], axis=0)
        hours = np.arange(len(cases[0]["hourly"]))
        grid = np.mean([[r.get("grid_purchase", max(r["user_load"] + r["charge"] - r["discharge"] - r["wind"], 0)) for r in case["hourly"]] for case in cases], axis=0)
        axes[0].plot(hours, grid, label="Policy grid imports")
        axes[0].plot(hours, average("user_load"), label="Policy response load")
        axes[0].plot(hours, average("baseline_load"), label="Baseline load")
        if all("reference_hourly" in c for c in cases):
            ref = np.mean([[max(r["user_load"] + r["charge"] - r["discharge"] - case["hourly"][i]["wind"], 0) for i, r in enumerate(case["reference_hourly"])] for case in cases], axis=0)
            axes[0].plot(hours, ref, label="Reference grid imports")
        axes[1].plot(hours, average("price"), label="Retail price")
        axes[1].axhline(np.mean(spec["user"]["price_min"]), linestyle="--", color="gray", label="Price bounds")
        axes[1].axhline(np.mean(spec["user"]["price_max"]), linestyle="--", color="gray")
        axes[2].plot(hours, average("charge"), label="Charge")
        axes[2].plot(hours, average("discharge"), label="Discharge")
        axes[3].plot(hours, average("soc"), label="SOC energy")
        axes[3].axhline(spec["storage"]["soc_min"], linestyle="--", color="gray", label="SOC minimum")
        for ax, ylabel in zip(axes, ("Power", "Price", "Power", "Stored energy")):
            ax.legend()
            ax.set_xlabel("Hour")
            ax.set_ylabel(ylabel + " (source units)")
            ax.grid(alpha=.2)
        fig.suptitle("Final development validation: mean hourly curves\nIndividual daily peaks may differ from the mean curve's peak")
        fig.savefig(out / "hourly_dispatch.png", dpi=160)
        plt.close(fig)
    return "Static PNG plots written."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--evaluation", type=Path, nargs="+")
    parser.add_argument("--environment", type=Path, help="Defaults to the run's frozen source/data/environment.json")
    parser.add_argument("--training", type=Path, help="Optional training.csv for standalone evaluation inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--plots", action="store_true", help="Generate PNGs if Matplotlib is installed")
    args = parser.parse_args()
    files = sorted(args.run_dir.glob("evaluation_*.json")) if args.run_dir else args.evaluation
    if not files:
        parser.error("No evaluation JSON files found")
    spec_path = args.environment or (args.run_dir / "source/data/environment.json" if args.run_dir else None)
    if spec_path is None:
        parser.error("--environment is required with --evaluation")
    spec = json.loads(spec_path.read_text("utf-8"))
    evaluations, report, case_rows, hour_rows, eval_rows = [], [], [], [], []
    for path in files:
        payload = json.loads(path.read_text("utf-8"))
        match = re.search(r"evaluation_(\d+)\.json$", path.name)
        episode = int(match.group(1)) if match else None
        analysis = analyze_evaluation(payload, spec)
        evaluations.append((episode, payload))
        report.append(dict(source=str(path.resolve()), episode=episode, analysis=analysis))
        eval_rows.append(dict(source=path.name, episode=episode, n=analysis["n"], checks_passed=analysis["checks_passed"],
                              **{k: v["mean"] for k, v in analysis["aggregate"].items()}))
        for i, case in enumerate(analysis["cases"]):
            case_rows.append(dict(source=path.name, episode=episode, case_index=i, **_flatten(case)))
            original = payload["cases"][i]
            for role, key in (("policy", "hourly"), ("reference", "reference_hourly")):
                for j, row in enumerate(original.get(key, [])):
                    exported = dict(row)
                    if role == "reference":
                        for k in ("wind", "baseline_load", "grid_price"):
                            if k in original["hourly"][j]:
                                exported.setdefault(k, original["hourly"][j][k])
                        exported.setdefault("grid_purchase", max(row["user_load"] + row["charge"] - row["discharge"] - exported["wind"], 0))
                    hour_rows.append(dict(source=path.name, episode=episode, case_index=i, role=role, **exported))
    training_path = args.training or (args.run_dir / "training.csv" if args.run_dir else None)
    learning = None
    if training_path and training_path.exists():
        with training_path.open(encoding="utf-8-sig", newline="") as stream:
            training = list(csv.DictReader(stream))
        if training:
            learning = analyze_learning(training, args.window)
    args.output.mkdir(parents=True, exist_ok=False)
    _write_csv(args.output / "evaluation_metrics.csv", eval_rows)
    _write_csv(args.output / "case_metrics.csv", case_rows)
    _write_csv(args.output / "hourly.csv", hour_rows)
    if learning:
        _write_csv(args.output / "learning_curves.csv", learning["rows"])
    plot_status = _plots(args.output, evaluations, report, learning, spec) if args.plots else "Not requested."
    summary = dict(definitions=DEFINITIONS, environment_source=str(spec_path.resolve()),
                   evaluations=report, learning=learning, plots=plot_status)
    (args.output / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps(dict(output=str(args.output.resolve()), evaluations=len(report),
                          all_physical_checks_passed=all(r["analysis"]["checks_passed"] for r in report),
                          training_episodes=learning["episodes"] if learning else 0, plots=plot_status), ensure_ascii=False))


if __name__ == "__main__":
    main()
