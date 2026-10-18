"""Read-only summary of completed runs/sac_revision_* profit-only experiments.

Development evaluations and optional holdout_108.json are written to separate
tables. Incomplete runs are listed as skipped; completed but inconsistent runs
are listed as invalid and excluded. All changes use actual minus baseline:
positive user/grid change means increased consumption/imports, not reduction.
No training, source copying or plotting is performed. Existing summary outputs
may be overwritten; original run files are never written.
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .analyze_price_response import analyze_case
else:
    from analyze_price_response import analyze_case


ROOT = Path(__file__).resolve().parents[1]
REWARD_DESCRIPTION = "original profit multiplied by reward_scale"
DEFINITIONS = {
    "splits": "development.csv uses each run's final 27-case development evaluation; holdout.csv uses only separately saved holdout_108.json. File names do not independently establish an untouched selection history.",
    "change_sign": "All user/grid changes are actual minus the stated baseline. Positive means increase; negative means reduction.",
    "low_high_hours": "Fixed bottom/top quartiles of the original profile baseline_load; not policy-dependent labels. TOU groups come from the original profile retail_price.",
    "user_change": "user_load - baseline_load, before wind/storage. This measures response relative to original demand, not a responsive tariff reference.",
    "grid_change": "actual grid - max(baseline_load-wind,0), using identical realized wind. Demand and storage contributions sum to the total, including zero-import clipping.",
    "price_correlations": "Pearson calculated within each scenario, then mean and positive fraction across valid cases. Neither statistic guarantees every scenario.",
    "weighted_prices": "Actual physical charge/discharge energy weights the realized wholesale price within each case; table reports the mean across valid cases. Positive price spread alone is not a profitability guarantee.",
    "storage_margin": "Grid-cost saving versus the same realized price/response path with zero storage minus throughput cost; no terminal SOC valuation.",
    "settlement_precision": "Storage margin uses the saved settlement totals. An independent float64 reconstruction is checked within a bound derived from float32 grid-cost rounding; its maximum absolute discrepancy is reported.",
    "validation_scope": "Frozen source hashes, effective input snapshots, complete/training counts, aggregate training returns, and all saved hourly policy learning rewards in the selected final development/holdout files are checked. Training hourly trajectories are not saved, so their individual hourly rewards cannot be audited retrospectively.",
    "inference": "These are sampled scenarios and saved trained models. Runs may change several permitted settings, so the table alone does not isolate a single causal factor or establish multi-seed robustness.",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_valid(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def fraction(values, predicate):
    valid = [v for v in values if v is not None]
    return float(np.mean([predicate(v) for v in valid])) if valid else None


def validate_run(path):
    cfg = read_json(path / "config.json")
    complete = read_json(path / "complete.json")
    manifest = read_json(path / "manifest.json")
    require(manifest.get("version") == "sac-2-profit-only", "Expected profit-only run version")
    require(manifest.get("reward") == REWARD_DESCRIPTION, "Manifest reward description differs")
    require(not any(k in cfg for k in ("reward", "shape_weight", "peak_weight")), "Retired reward configuration present")
    episodes = cfg["episodes"]
    require(isinstance(episodes, int) and not isinstance(episodes, bool) and episodes > 0, "Invalid episode count")
    require(complete["episodes"] == episodes and complete["steps"] == episodes * 24, "Complete/config episode or step mismatch")
    require(not (path / "failure.json").exists(), "Failure marker present")
    hashes = manifest["hashes"]
    require(bool(hashes), "Empty source hash manifest")
    for name, digest in hashes.items():
        frozen = path / "source" / name
        require(frozen.is_file() and sha256(frozen) == digest, "Frozen source hash mismatch: " + name)
    with (path / "training.csv").open(encoding="utf-8-sig", newline="") as stream:
        training = list(csv.DictReader(stream))
    require(len(training) == episodes and [int(r["episode"]) for r in training] == list(range(1, episodes + 1)), "Training rows must cover exactly episodes 1..N")
    require(int(training[-1]["updates"]) == complete["updates"], "Training/complete update mismatch")
    scale = cfg["reward_scale"]
    require(np.isfinite(scale) and scale > 0, "Invalid reward scale")
    for row in training:
        profit, learned = float(row["profit"]), float(row["learning_return"])
        require(np.isfinite([profit, learned]).all() and np.isclose(learned, profit * scale, rtol=1e-12, atol=1e-8),
                "Training return is inconsistent with scaled profit")
    environment = read_json(path / "effective_environment.json")
    profile = read_json(path / "effective_profile.json")
    generator = read_json(path / "effective_generator.json")
    expected = copy.deepcopy(read_json(path / "source/data/environment.json"))
    expected["demand"].update(cfg["response_parameters"])
    expected["storage"].update(cfg["storage_parameters"])
    require(environment == expected, "Effective environment differs from frozen inputs plus configured parameters")
    original_profile = read_json(path / "source/data/profile.json")
    original_profile = {k: np.asarray(v, dtype=np.float32).tolist() for k, v in original_profile.items()}
    require(profile == original_profile, "Effective profile differs from frozen original profile")
    require(generator["target_penetration"] == cfg["renewable_penetration"], "Effective renewable penetration differs")
    return dict(path=path, config=cfg, complete=complete, environment=environment, profile=profile,
                generator=generator, frozen_source_files_checked=len(hashes))


def configuration_columns(run):
    cfg, env = run["config"], run["environment"]
    row = dict(run=run["path"].name, **{key: cfg[key] for key in
               ("seed", "episodes", "observation", "reward_scale", "hidden_dim", "actor_lr", "critic_lr",
                "alpha_lr", "initial_alpha", "target_entropy", "gamma", "target_tau", "batch_size",
                "warmup_steps", "replay_size", "threads", "renewable_penetration", "scenario_mode", "response_rng")})
    row.update({"response_" + k: env["demand"][k] for k in ("curvature_mean", "curvature_std", "min_curvature", "demand_noise_std")})
    row.update({"storage_" + k: env["storage"][k] for k in
                ("soc_min", "soc_max", "soc_init", "p_ch_max", "p_dis_max", "eta_ch", "eta_dis", "c_es")})
    row["storage_usable_energy"] = env["storage"]["soc_max"] - env["storage"]["soc_min"]
    row["training_updates"] = run["complete"]["updates"]
    row["training_wall_seconds"] = run["complete"]["wall_seconds"]
    return row


def summarize_evaluation(run, evaluation_path, split):
    payload = read_json(evaluation_path)
    cfg, env, profile = run["config"], run["environment"], run["profile"]
    expected_count = 27 if split == "development" else 108
    require(payload["n"] == expected_count and len(payload["cases"]) == expected_count, f"{split}: expected {expected_count} cases")
    require(payload.get("deterministic") is True, "Expected deterministic evaluation")
    metadata = payload["metadata"]
    for key, value in (("environment", env), ("profile", profile), ("generator", run["generator"]), ("config", cfg)):
        require(metadata.get(key) == value, f"Evaluation metadata {key} differs from effective inputs")
    require(metadata.get("reward") == REWARD_DESCRIPTION, "Evaluation reward metadata differs")
    diagnostics, margin_reconstruction_errors = [], []
    for index, case in enumerate(payload["cases"]):
        require(case["case"] == index and case["scenario_seed"] == payload["seed"] + index, "Evaluation case identity/order mismatch")
        response_seed = 42 if cfg["response_rng"] == "paired_fixed" else case["scenario_seed"] + 700000
        require(case["response_seed"] == response_seed, "Evaluation response seed mismatch")
        require(len(case["hourly"]) == 24, "Incomplete hourly trajectory")
        for row in case["hourly"]:
            require(np.isfinite([row["profit"], row["learning_reward"]]).all()
                    and row["learning_reward"] == row["profit"] * cfg["reward_scale"],
                    "Hourly learning reward is not exactly profit * reward_scale")
        checked = analyze_case(case, env, profile, metadata)
        require(checked["checks"]["all_ok"] and checked.get("reference_policy", {}).get("checks", {}).get("all_ok", True), "Hourly physical checks failed")
        recomputed_profit = sum(row["profit"] for row in case["hourly"])
        require(np.isclose(case["profit"], recomputed_profit, rtol=1e-12, atol=1e-8), "Daily profit differs from hourly total")
        settled_margin = sum(r["storage_marginal_profit_reward"] for r in case["hourly"])
        require(case["storage_net"] == settled_margin, "Daily storage margin differs from hourly settlements")
        error = abs(settled_margin - checked["storage"]["same_price_storage_margin"])
        # The environment settles np.float32 grid-price products. Reconstructing
        # those same powers in float64 can differ by a few thousandths per day.
        cost_magnitude = sum(abs(r["grid_price"]) * (r["grid"] + r["same_price_no_storage_grid"])
                             for r in checked["hourly"]) * env["params"]["dt"]
        require(error <= np.finfo(np.float32).eps * cost_magnitude + 1e-8,
                "Saved storage margin differs from the physical counterfactual beyond float32 settlement rounding")
        margin_reconstruction_errors.append(error)
        grid = np.array([r["grid"] for r in checked["hourly"]])
        for key, value in (("peak", grid.max()), ("peak_valley", np.ptp(grid)), ("std", grid.std())):
            require(np.isclose(case["curves"]["grid"][key], value, rtol=1e-12, atol=1e-8), "Saved daily grid metric differs from hourly inputs")
        diagnostics.append(checked)
    row = dict(configuration_columns(run), split=split, evaluation_seed=payload["seed"], cases=expected_count,
               profit_mean=float(np.mean([c["profit"] for c in payload["cases"]])),
               reference_profit_mean=float(np.mean([c["reference_profit"] for c in payload["cases"]])),
               storage_margin_mean=float(np.mean([c["storage_net"] for c in payload["cases"]])),
               storage_margin_positive_fraction=fraction([c["storage_net"] for c in payload["cases"]], lambda x: x > 0),
               storage_margin_recompute_max_abs_difference=max(margin_reconstruction_errors),
               physical_checks_passed=True, reward_checks_passed=True, effective_metadata_matched=True,
               frozen_source_files_checked=run["frozen_source_files_checked"])
    for metric in ("peak", "peak_valley", "std"):
        row["grid_" + metric + "_mean"] = float(np.mean([c["curves"]["grid"][metric] for c in payload["cases"]]))
    for target in ("realized_wholesale", "original_tou"):
        values = [c["prices"]["correlation_with_" + target]["pearson"] for c in diagnostics]
        row["price_" + target + "_pearson_mean"] = mean_valid(values)
        row["price_" + target + "_positive_fraction"] = fraction(values, lambda x: x > 0)
        row["price_" + target + "_valid_cases"] = sum(x is not None for x in values)
    for tariff in np.unique(profile["retail_price"]):
        label = str(float(tariff))
        row["tou_" + label.replace(".", "_") + "_policy_price_mean"] = float(np.mean([c["prices"]["tou_bands"][label]["retail_price_mean"] for c in diagnostics]))
    row["tou_means_ordered_fraction"] = fraction([c["prices"]["tou_means_nondecreasing"] for c in diagnostics], bool)
    for key in ("weighted_charge_wholesale_price", "weighted_discharge_wholesale_price", "discharge_minus_charge_price",
                "charge_energy", "discharge_energy", "final_soc", "soc_change"):
        row[key + "_mean"] = mean_valid([c["storage"][key] for c in diagnostics])
    row["discharge_minus_charge_price_positive_fraction"] = fraction([c["storage"]["discharge_minus_charge_price"] for c in diagnostics], lambda x: x > 0)
    for band in ("low", "high"):
        values = [-c["bands"]["nominal_load"][band]["response_reduction_mean"]
                  if c["bands"]["nominal_load"][band]["response_reduction_mean"] is not None else None for c in diagnostics]
        row[band + "_load_user_change_mean"] = mean_valid(values)
        row[band + "_load_user_increase_fraction"] = fraction(values, lambda x: x > 0)
        row[band + "_load_user_decrease_fraction"] = fraction(values, lambda x: x < 0)
        for name, source in (("grid_change", "total_grid_effect_mean"), ("demand_grid_change", "demand_grid_effect_mean"), ("storage_grid_change", "storage_grid_effect_mean")):
            row[band + "_load_" + name + "_mean"] = mean_valid([c["bands"]["nominal_load"][band][source] for c in diagnostics])
    row["season_metadata_cases"] = sum(c["context"]["season"] != "unknown" for c in diagnostics)
    row["source_levels_metadata_cases"] = sum(all(c["context"][k + "_level"] != "unknown" for k in ("load", "wind", "price")) for c in diagnostics)
    return row


def summarize_runs(runs_root, pattern="sac_revision_*"):
    result = dict(definitions=DEFINITIONS, development=[], holdout=[], skipped=[], invalid_runs=[], configurations={})
    for path in sorted(runs_root.glob(pattern)):
        if not path.is_dir():
            continue
        if not (path / "complete.json").exists():
            result["skipped"].append(dict(run=path.name, reason="No complete.json; run is incomplete"))
            continue
        try:
            run = validate_run(path)
            final = path / f"evaluation_{run['config']['episodes']:05d}.json"
            development = summarize_evaluation(run, final, "development")
            holdout = summarize_evaluation(run, path / "holdout_108.json", "holdout") if (path / "holdout_108.json").exists() else None
        except (ValueError, KeyError, TypeError, OSError) as exc:
            result["invalid_runs"].append(dict(run=path.name, error=f"{type(exc).__name__}: {exc}"))
            continue
        result["development"].append(development)
        if holdout is not None:
            result["holdout"].append(holdout)
        result["configurations"][path.name] = run["config"]
    return result


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["run", "split", "cases"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--pattern", default="sac_revision_*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_runs(args.runs_root.resolve(), args.pattern)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("development", "holdout"):
        write_csv(args.output / (split + ".csv"), report[split])
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps(dict(output=str(args.output.resolve()), development_runs=len(report["development"]),
                          holdout_runs=len(report["holdout"]), skipped=report["skipped"], invalid_runs=report["invalid_runs"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
