"""Diagnose saved retail prices, physical storage dispatch and demand response.

All labels are exogenous: original profile TOU bands and profile wholesale/load
quartiles, plus separately named realized scenario quartiles. No actor output is
used to define a low/high period. These diagnostics never alter rewards or actions.

Input: old/new SAC evaluation JSON with cases[].hourly; environment and original
profile JSON must be supplied explicitly. New metadata.environment/profile
snapshots, when present, must agree with those inputs. Optional case.levels and
case.season support joint scenario/season summaries; missing labels stay unknown.

Example:
  python tools/analyze_price_response.py --evaluation runs/example/holdout_108.json \
      --environment runs/example/effective_environment.json \
      --profile runs/example/effective_profile.json --output checks/price_response
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .analyze_dispatch import analyze_case as analyze_dispatch_case
else:
    from analyze_dispatch import analyze_case as analyze_dispatch_case


DEFINITIONS = {
    "purpose": "Diagnostic measurements only: no reward changes, constraints, tariff targets or policy rules.",
    "labels": "TOU bands and nominal wholesale/load quartiles come from the supplied original profile. Realized quartiles use only a scenario's exogenous inputs, never prices/actions chosen by the policy.",
    "quantile_ties": "Low <= 25th percentile, high >= 75th percentile; if these thresholds coincide, all hours are middle to avoid overlapping low/high labels.",
    "correlations": "Pearson and tie-aware Spearman are calculated separately within each scenario. Aggregate means and positive-case fractions do not guarantee alignment in each scenario.",
    "response": "baseline_load - user_load: positive is demand reduction; negative is increased consumption. This is relative to original demand, not an already-responsive nominal-tariff policy.",
    "counterfactual": "Replay the same realized retail-price path and same demand response with zero storage: max(user_load - wind, 0). This holds actions/prices fixed, not a re-optimized or re-sampled policy rollout.",
    "decomposition": "baseline_grid=max(baseline_load-wind,0); demand_effect=same_price_no_storage_grid-baseline_grid; storage_effect=actual_grid-same_price_no_storage_grid. Their sum equals actual_grid-baseline_grid, including zero-import clipping.",
    "storage_alignment": "Charge/discharge are actual physical powers. Energy-weighted realized wholesale prices describe timing; a positive discharge-minus-charge price spread alone does not establish profitable arbitrage.",
    "storage_margin": "Grid-cost saving versus same-price/no-storage minus stated throughput cost; includes zero-export clipping and conversion losses embodied in dispatch, but no terminal SOC valuation.",
    "context": "Report season × load/wind/price-level strata only from explicit metadata. Missing season is unknown; temperature alone does not establish season. Within-scenario, cross-scenario and hourly-mean patterns are distinct.",
    "units": "Source power/price units; energy is power times dt. No currency or physical calibration is inferred.",
}


def _vector(values, name, length=None):
    try:
        a = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite vector") from exc
    if a.ndim != 1 or not a.size or not np.isfinite(a).all() or (length is not None and len(a) != length):
        raise ValueError(f"{name} must be a finite vector of the expected length")
    return a


def quartile_masks(values):
    a = _vector(values, "quartile input")
    low, high = np.quantile(a, [.25, .75])
    if low == high:
        return dict(low=np.zeros(a.size, bool), middle=np.ones(a.size, bool), high=np.zeros(a.size, bool))
    return dict(low=a <= low, middle=(a > low) & (a < high), high=a >= high)


def _ranks(a):
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=float)
    sorted_a = a[order]
    start = 0
    while start < a.size:
        end = start + 1
        while end < a.size and sorted_a[end] == sorted_a[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def correlation(a, b):
    """Return undefined correlations as null, including constant price paths."""
    a = _vector(a, "correlation input")
    b = _vector(b, "correlation input", len(a))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return dict(pearson=None, spearman=None)
    return dict(pearson=float(np.corrcoef(a, b)[0, 1]),
                spearman=float(np.corrcoef(_ranks(a), _ranks(b))[0, 1]))


def _weighted_mean(values, weights):
    return float(np.dot(values, weights) / weights.sum()) if weights.sum() > 1e-12 else None


def _mean(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def _context(case, payload_metadata):
    levels = case.get("levels", case.get("scenario_levels", {}))
    if isinstance(levels, (list, tuple)):
        if len(levels) != 3:
            raise ValueError("Case levels must have load, wind and price entries")
        levels = dict(zip(("load", "wind", "price"), levels))
    if not isinstance(levels, dict):
        raise ValueError("Case levels must be a mapping or a three-item sequence")
    def label(value):
        return "unknown" if value is None or value == "" else str(value)
    return dict(season=label(case.get("season", payload_metadata.get("season", "unknown"))),
                **{key + "_level": label(levels.get(key, "unknown")) for key in ("load", "wind", "price")})


def _band_summary(mask, data, dt):
    """The same exogenous mask is used for prices, response and all grid components."""
    return dict(
        hours=np.flatnonzero(mask).tolist(), count=int(mask.sum()),
        retail_price_mean=_mean(data["price"], mask), wholesale_price_mean=_mean(data["grid_price"], mask),
        baseline_load_mean=_mean(data["baseline_load"], mask), user_load_mean=_mean(data["user_load"], mask),
        response_reduction_mean=_mean(data["response_reduction"], mask),
        grid_mean=_mean(data["grid"], mask), baseline_grid_mean=_mean(data["baseline_grid"], mask),
        demand_grid_effect_mean=_mean(data["demand_effect"], mask),
        storage_grid_effect_mean=_mean(data["storage_effect"], mask),
        total_grid_effect_mean=_mean(data["total_effect"], mask),
        charge_energy=float(data["charge"][mask].sum() * dt),
        discharge_energy=float(data["discharge"][mask].sum() * dt),
    )


def analyze_case(case, environment, profile, payload_metadata=None):
    """Return one scenario's price relationships and physical attribution."""
    rows = case["hourly"]
    horizon = len(profile["retail_price"])
    if len(rows) != horizon:
        raise ValueError("Hourly trajectory and original profile lengths differ")
    data = {key: _vector([r[key] for r in rows], key, horizon) for key in
            ("baseline_load", "wind", "grid_price", "user_load", "price", "charge", "discharge")}
    tou = _vector(profile["retail_price"], "profile retail_price", horizon)
    nominal_wholesale = _vector(profile["grid_price"], "profile grid_price", horizon)
    nominal_load = _vector(profile["baseline_load"], "profile baseline_load", horizon)
    physical = analyze_dispatch_case(case, environment)
    dt = float(environment["params"]["dt"])
    data["baseline_grid"] = np.maximum(data["baseline_load"] - data["wind"], 0)
    data["same_price_no_storage_grid"] = np.maximum(data["user_load"] - data["wind"], 0)
    expected_grid = np.maximum(data["user_load"] + data["charge"] - data["discharge"] - data["wind"], 0)
    data["grid"] = np.array([r.get("grid_purchase", expected_grid[i]) for i, r in enumerate(rows)], dtype=float)
    data["response_reduction"] = data["baseline_load"] - data["user_load"]
    data["demand_effect"] = data["same_price_no_storage_grid"] - data["baseline_grid"]
    data["storage_effect"] = data["grid"] - data["same_price_no_storage_grid"]
    data["total_effect"] = data["grid"] - data["baseline_grid"]

    tou_values = np.unique(tou)
    tou_bands = {str(float(value)): _band_summary(tou == value, data, dt) for value in tou_values}
    means = np.array([tou_bands[str(float(v))]["retail_price_mean"] for v in tou_values])
    masks = dict(nominal_wholesale=quartile_masks(nominal_wholesale),
                 realized_wholesale=quartile_masks(data["grid_price"]),
                 nominal_load=quartile_masks(nominal_load),
                 realized_load=quartile_masks(data["baseline_load"]))
    bands = {name: {band: _band_summary(mask, data, dt) for band, mask in partition.items()}
             for name, partition in masks.items()}
    charge, discharge = data["charge"], data["discharge"]
    charge_price = _weighted_mean(data["grid_price"], charge)
    discharge_price = _weighted_mean(data["grid_price"], discharge)
    cost_saving = float(np.dot(data["same_price_no_storage_grid"] - data["grid"], data["grid_price"]) * dt)
    throughput_cost = float((charge + discharge).sum() * dt * environment["storage"]["c_es"])
    storage = dict(charge_energy=float(charge.sum() * dt), discharge_energy=float(discharge.sum() * dt),
                   weighted_charge_wholesale_price=charge_price, weighted_discharge_wholesale_price=discharge_price,
                   discharge_minus_charge_price=discharge_price - charge_price if charge_price is not None and discharge_price is not None else None,
                   both_charge_and_discharge=bool(charge.sum() > 1e-12 and discharge.sum() > 1e-12),
                   same_price_grid_cost_saving=cost_saving, throughput_cost=throughput_cost,
                   same_price_storage_margin=cost_saving - throughput_cost,
                   initial_soc=physical["storage"]["initial_soc"], final_soc=physical["storage"]["final_soc"],
                   soc_change=physical["storage"]["soc_change"])
    for partition in ("nominal_wholesale", "realized_wholesale"):
        for direction, power in (("charge", charge), ("discharge", discharge)):
            for band in ("low", "high"):
                storage[f"{partition}_{band}_{direction}_fraction"] = (
                    float(power[masks[partition][band]].sum() / power.sum()) if power.sum() > 1e-12 else None)
    result = dict(
        identity={key: case[key] for key in ("case", "scenario_seed", "response_seed") if key in case},
        context=_context(case, payload_metadata or {}),
        prices=dict(correlation_with_realized_wholesale=correlation(data["price"], data["grid_price"]),
                    correlation_with_original_tou=correlation(data["price"], tou),
                    correlation_with_nominal_wholesale=correlation(data["price"], nominal_wholesale),
                    tou_means_nondecreasing=bool(np.all(np.diff(means) >= -1e-9)) if len(means) > 1 else None,
                    tou_high_minus_low_mean=float(means[-1] - means[0]) if len(means) > 1 else None,
                    tou_bands_ranked_by_policy_mean=tou_values[np.argsort(means, kind="stable")].tolist(),
                    tou_bands=tou_bands),
        storage=storage, bands=bands,
        response=dict(baseline_energy=float(data["baseline_load"].sum() * dt),
                      user_energy=float(data["user_load"].sum() * dt),
                      reduction_energy=float(data["response_reduction"].sum() * dt)),
        grid_decomposition=dict(baseline_energy=float(data["baseline_grid"].sum() * dt),
                                same_price_no_storage_energy=float(data["same_price_no_storage_grid"].sum() * dt),
                                actual_energy=float(data["grid"].sum() * dt),
                                demand_effect_energy=float(data["demand_effect"].sum() * dt),
                                storage_effect_energy=float(data["storage_effect"].sum() * dt),
                                total_effect_energy=float(data["total_effect"].sum() * dt),
                                max_additivity_error=float(np.abs(data["demand_effect"] + data["storage_effect"] - data["total_effect"]).max())),
        checks=physical["checks"],
        hourly=[dict(hour=i, original_tou=float(tou[i]), **{key: float(values[i]) for key, values in data.items()}) for i in range(horizon)],
    )
    # Reference summaries are optional and do not define original-demand response.
    if "reference" in physical:
        result["reference_policy"] = physical["reference"]
    return result


def _flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(_flatten(item, name + "."))
        elif item is None or isinstance(item, (int, float, bool)):
            result[name] = item
    return result


def summarize_cases(cases):
    """Summarize distributions of per-case metrics, never pool all hours for correlation."""
    flat = [_flatten({k: v for k, v in c.items() if k not in ("identity", "context", "checks", "reference_policy", "hourly")}) for c in cases]
    summary = {}
    for name in dict.fromkeys(key for row in flat for key in row):
        values = [row.get(name) for row in flat if row.get(name) is not None]
        if values and all(isinstance(v, bool) for v in values):
            summary[name] = dict(valid_cases=len(values), missing_cases=len(cases) - len(values), true_fraction=float(np.mean(values)))
        elif values:
            a = np.asarray(values, dtype=float)
            summary[name] = dict(valid_cases=len(values), missing_cases=len(cases) - len(values),
                                 mean=float(a.mean()), std=float(a.std()), minimum=float(a.min()),
                                 p10=float(np.quantile(a, .1)), median=float(np.median(a)), p90=float(np.quantile(a, .9)),
                                 maximum=float(a.max()), positive_fraction=float(np.mean(a > 0)))
        else:
            summary[name] = dict(valid_cases=0, missing_cases=len(cases), mean=None)
    return dict(cases=len(cases), metrics=summary)


def analyze_evaluation(payload, environment, profile):
    metadata = payload.get("metadata", {})
    for key, supplied in (("environment", environment), ("profile", profile)):
        if key in metadata and metadata[key] != supplied:
            raise ValueError(f"Supplied {key} differs from the evaluation's metadata snapshot")
    original = payload["cases"]
    if not original or ("n" in payload and payload["n"] != len(original)):
        raise ValueError("Evaluation case count is empty or inconsistent")
    cases = [analyze_case(case, environment, profile, metadata) for case in original]
    groups = {}
    for case in cases:
        key = json.dumps(case["context"], sort_keys=True)
        groups.setdefault(key, []).append(case)
    contexts = [dict(context=group[0]["context"], summary=summarize_cases(group)) for group in groups.values()]
    nominal = {key: np.asarray(profile[key], dtype=float) for key in ("retail_price", "grid_price", "baseline_load")}
    return dict(
        definitions=DEFINITIONS,
        fixed_labels=dict(tou_bands={str(float(v)): np.flatnonzero(nominal["retail_price"] == v).tolist() for v in np.unique(nominal["retail_price"])},
                          nominal_wholesale={k: np.flatnonzero(v).tolist() for k, v in quartile_masks(nominal["grid_price"]).items()},
                          nominal_load={k: np.flatnonzero(v).tolist() for k, v in quartile_masks(nominal["baseline_load"]).items()}),
        all_physical_checks_passed=all(c["checks"]["all_ok"] for c in cases),
        aggregate=summarize_cases(cases), contexts=contexts, cases=cases,
        context_coverage=dict(explicit_season_cases=sum(c["context"]["season"] != "unknown" for c in cases),
                              explicit_all_source_levels_cases=sum(all(c["context"][k + "_level"] != "unknown" for k in ("load", "wind", "price")) for c in cases)),
    )


def plot_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = report["cases"]
    def mean(key):
        return np.mean([[r[key] for r in c["hourly"]] for c in cases], axis=0)
    hours = np.arange(len(cases[0]["hourly"]))
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    ax = axes[0, 0]
    for key, label in (("price", "Policy retail"), ("original_tou", "Original TOU"), ("grid_price", "Wholesale")):
        ax.plot(hours, mean(key), label=label)
    ax.set(title="Mean hourly prices", xlabel="Hour", ylabel="Price (source units)")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for offset, key, label in ((0, "correlation_with_realized_wholesale", "Wholesale"), (1, "correlation_with_original_tou", "Original TOU")):
        values = [c["prices"][key]["pearson"] for c in cases if c["prices"][key]["pearson"] is not None]
        if values:
            ax.scatter(np.full(len(values), offset), values, alpha=.35, s=12)
            ax.scatter(offset, np.mean(values), marker="D", color="black", s=28)
    ax.set(title="Within-scenario Pearson correlations", xticks=[0, 1], xticklabels=["Wholesale", "Original TOU"], ylim=(-1.05, 1.05), ylabel="Each dot = one scenario; diamond = mean")
    ax.axhline(0, color="gray", linewidth=.7)
    ax = axes[1, 0]
    ax.plot(hours, mean("charge"), label="Actual charge")
    ax.plot(hours, -mean("discharge"), label="Actual discharge (negative)")
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set(title="Physical battery power", xlabel="Hour", ylabel="Power (source units)")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for key, label in (("baseline_load", "Original baseline load"), ("user_load", "After demand response"), ("grid", "Actual grid imports")):
        ax.plot(hours, mean(key), label=label)
    ax.set(title="Demand, response and grid imports", xlabel="Hour", ylabel="Power (source units)")
    ax.legend(fontsize=8)
    ax = axes[2, 0]
    for key, label in (("demand_effect", "Demand-response contribution"), ("storage_effect", "Storage contribution"), ("total_effect", "Total grid change")):
        ax.plot(hours, mean(key), label=label)
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set(title="Grid change from baseline with same wind", xlabel="Hour", ylabel="Power change (source units)")
    ax.legend(fontsize=8)
    ax = axes[2, 1]
    for offset, band in enumerate(("low", "high")):
        for shift, key, label, color in ((-.2, "response_reduction_mean", "User demand change", "#357bc1"), (0, "storage_grid_effect_mean", "Storage grid change", "#d36a25"), (.2, "total_grid_effect_mean", "Total grid change", "#14866d")):
            values = [c["bands"]["nominal_load"][band][key] for c in cases]
            values = [v for v in values if v is not None]
            if key == "response_reduction_mean":
                # All plotted changes use actual minus baseline: positive is
                # an increase. The stored reduction diagnostic uses the inverse.
                values = [-v for v in values]
            if values:
                ax.bar(offset + shift, np.mean(values), .18, color=color, label=label if offset == 0 else None)
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set(title="Fixed original-load quartile hours", xticks=[0, 1], xticklabels=["Low-load hours", "High-load hours"], ylabel="Mean power change (source units)")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("Price/response diagnostics: scenario distributions and hourly means\nAggregate patterns do not guarantee every scenario; no reward or control rule is changed", fontsize=13)
    fig.savefig(output / "price_response.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: getattr(args, key) for key in ("evaluation", "environment", "profile")}
    inputs = {key: json.loads(path.read_text("utf-8")) for key, path in paths.items()}
    report = analyze_evaluation(inputs["evaluation"], inputs["environment"], inputs["profile"])
    report["sources"] = {key: dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for key, path in paths.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "price_response.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    plot_report(report, args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), cases=report["aggregate"]["cases"],
                          all_physical_checks_passed=report["all_physical_checks_passed"],
                          context_coverage=report["context_coverage"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
