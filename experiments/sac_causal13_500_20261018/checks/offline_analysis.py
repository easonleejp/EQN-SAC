"""Compare two saved fixed-day tests; never import or execute a policy/environment.

Run only after the single causal13 test has been saved:
    .venv/bin/python checks/sac_causal13_500/offline_analysis.py

Inputs and output are resolved from this file, independent of the working directory.
Only offline_comparison.json is written. Prices/profits retain their recorded units;
under this study's convention they are Australian cents/kWh and Australian cents.
This paired single-day comparison is not an unseen-scenario/generalization test.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = {"before": ROOT / "runs/sac_current6_tuned_500",
        "after": ROOT / "runs/sac_causal13_500"}
PRICE_HOURS = list(range(10, 15))
RESPONSE_HOURS = list(range(11, 16))
EARLY_DISCHARGE_HOURS = list(range(8, 16))
EVENING_DISCHARGE_HOURS = list(range(17, 21))
NOISE_KEYS = ("curvature_std", "demand_noise_std", "participation_noise_std")
JSON_NAMES = ("config.json", "complete.json", "manifest.json", "test_1ep.json",
              "effective_environment.json", "effective_profile.json",
              "effective_generator.json")
VECTOR_KEYS = ("hour", "price", "grid_price", "baseline_load", "user_load", "wind",
               "grid_purchase", "charge", "discharge", "initial_soc", "soc",
               "profit", "learning_reward", "total_response",
               "storage_marginal_profit_reward")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def close(actual, expected, message, atol=1e-5, rtol=0):
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), message)


def read_run(path):
    data = {name: json.loads((path / name).read_text("utf-8")) for name in JSON_NAMES}
    cfg, complete, result = (data[name] for name in
                             ("config.json", "complete.json", "test_1ep.json"))
    env, profile, generator = (data[f"effective_{name}.json"] for name in
                               ("environment", "profile", "generator"))
    require(cfg["episodes"] == complete["episodes"] == 500, f"{path}: incomplete 500 episodes")
    require(cfg["scenario_mode"] == "deterministic_day", f"{path}: not a fixed day")
    require(cfg["evaluation_every"] == 0 and not list(path.glob("evaluation_*.json")),
            f"{path}: internal evaluations must be disabled")
    require(not (path / "failure.json").exists(), f"{path}: failure record exists")
    require(result["n"] == len(result["cases"]) == 1 and result["deterministic"] is True,
            f"{path}: expected exactly one deterministic-action test case")
    require(result["metadata"]["reward"] == data["manifest.json"]["reward"]
            == "original profit multiplied by reward_scale",
            f"{path}: reward definition metadata differs from the original profit objective")
    for key, expected in (("config", cfg), ("environment", env),
                          ("profile", profile), ("generator", generator)):
        require(result["metadata"][key] == expected, f"{path}: test {key} metadata differs")
    require(all(cfg["response_parameters"].get(k) == env["demand"][k] == 0
                for k in NOISE_KEYS), f"{path}: response noise is not explicitly zero")
    require(env["params"]["dt"] == generator["dt"] == 1, f"{path}: expected one-hour steps")
    case = result["cases"][0]
    require(result["seed"] == case["scenario_seed"] == cfg["reference_seed"],
            f"{path}: fixed-day seed mismatch")
    require(case["levels"] == dict.fromkeys(("load", "wind", "price"), "deterministic"),
            f"{path}: case levels are not deterministic")
    require(cfg["response_rng"] == "paired_fixed" and case["response_seed"] == env["response_seed"],
            f"{path}: response seed mismatch")
    rows = case["hourly"]
    require(len(rows) == 24, f"{path}: expected 24 saved hours")
    values = {key: np.asarray([r[key] for r in rows], dtype=float) for key in VECTOR_KEYS}
    require(all(np.isfinite(v).all() for v in values.values()), f"{path}: nonfinite hourly values")
    require(np.array_equal(values["hour"], np.arange(24)), f"{path}: hours must be 0..23")
    for key in ("baseline_load", "grid_price"):
        require(np.array_equal(values[key], profile[key]), f"{path}: {key} differs from profile")
    require(np.all(values["wind"] == values["wind"][0]), f"{path}: fixed wind is not constant")
    close(values["wind"].sum() / values["baseline_load"].sum(), cfg["renewable_penetration"],
          f"{path}: renewable share mismatch", atol=1e-6)

    with (path / "training.csv").open(encoding="utf-8", newline="") as stream:
        training = list(csv.DictReader(stream))
    require([int(r["episode"]) for r in training] == list(range(1, 501)),
            f"{path}: expected all 500 training rows")
    require(complete["steps"] == 24 * len(training) == 12000,
            f"{path}: expected 12000 environment steps")
    updates = (12000 - max(1, cfg["warmup_steps"], cfg["batch_size"]) + 1) * cfg["updates_per_step"]
    require(complete["updates"] == int(training[-1]["updates"]) == updates,
            f"{path}: update count mismatch")
    profit = np.asarray([float(r["profit"]) for r in training])
    reward = np.asarray([float(r["learning_return"]) for r in training])
    require(np.isfinite(profit).all() and np.isfinite(reward).all(), f"{path}: nonfinite training return")
    close(reward, profit * cfg["reward_scale"], f"{path}: training reward differs from profit scale",
          atol=1e-8, rtol=1e-11)
    close(values["learning_reward"], values["profit"] * cfg["reward_scale"],
          f"{path}: test reward differs from profit scale", atol=1e-8, rtol=1e-11)
    close(values["profit"].sum(), case["profit"], f"{path}: hourly profit total mismatch")
    close(values["storage_marginal_profit_reward"].sum(), case["storage_net"],
          f"{path}: hourly storage marginal profit mismatch")

    storage = env["storage"]
    charge, discharge = values["charge"], values["discharge"]
    require(np.all(charge >= 0) and np.all(discharge >= 0) and np.all(charge * discharge == 0),
            f"{path}: invalid/simultaneous charge and discharge")
    require(np.all(charge <= storage["p_ch_max"] + 1e-5)
            and np.all(discharge <= storage["p_dis_max"] + 1e-5), f"{path}: power limit exceeded")
    require(np.all(values["soc"] >= storage["soc_min"] - 1e-5)
            and np.all(values["soc"] <= storage["soc_max"] + 1e-5), f"{path}: SOC limit exceeded")
    close(values["initial_soc"][0], storage["soc_init"], f"{path}: initial SOC mismatch")
    close(values["initial_soc"][1:], values["soc"][:-1], f"{path}: discontinuous SOC")
    close(values["soc"], values["initial_soc"] * storage["eta_self"]
          + charge * storage["eta_ch"] - discharge / storage["eta_dis"],
          f"{path}: SOC energy balance mismatch")
    close(values["grid_purchase"], np.maximum(values["user_load"] - values["wind"]
          + charge - discharge, 0), f"{path}: grid energy balance mismatch")
    # Class loads/responses are reduced in float32; summed totals can differ below 0.002 kW.
    close(values["baseline_load"] - values["user_load"], values["total_response"],
          f"{path}: recorded response mismatch", atol=.002)
    close(values["soc"][-1], case["final_soc"], f"{path}: final SOC mismatch")
    for key in ("charge", "discharge"):
        close(values[key].sum(), case[key], f"{path}: {key} total mismatch")

    sources = [fingerprint(path / name) for name in (*JSON_NAMES, "training.csv")]
    for relative, expected in data["manifest.json"]["hashes"].items():
        item = fingerprint(path / "source" / relative)
        require(item["sha256"] == expected, f"{path}: frozen source differs: {relative}")
        sources.append(item)
    return dict(path=path, cfg=cfg, env=env, profile=profile, generator=generator,
                case=case, values=values, complete=complete, training=training, sources=sources)


def pearson(x, y):
    return None if np.ptp(x) == 0 or np.ptp(y) == 0 else float(np.corrcoef(x, y)[0, 1])


def curve(values):
    change = np.diff(values)
    index = int(np.argmax(np.abs(change)))
    return dict(peak=float(values.max()), valley=float(values.min()),
                range=float(np.ptp(values)), mean=float(values.mean()), energy=float(values.sum()),
                std=float(values.std()), rms_level=float(np.sqrt(np.mean(values ** 2))),
                rms_ramp=float(np.sqrt(np.mean(change ** 2))),
                max_abs_jump=float(abs(change[index])), max_jump_signed=float(change[index]),
                max_jump_hours=[index, index + 1])


def window_response(values, hours):
    base, user, wind, grid = (values[k][hours] for k in
                              ("baseline_load", "user_load", "wind", "grid_purchase"))
    baseline_grid = np.maximum(base - wind, 0)
    no_storage_grid = np.maximum(user - wind, 0)
    return dict(hours=hours, baseline_energy=float(base.sum()), user_energy=float(user.sum()),
                user_change_mean=float(np.mean(user - base)),
                user_change_energy=float(np.sum(user - base)),
                user_change_percent=float(100 * (user.sum() / base.sum() - 1)),
                reduction_percent=float(100 * (1 - user.sum() / base.sum())),
                baseline_grid_energy=float(baseline_grid.sum()), grid_energy=float(grid.sum()),
                grid_change_mean=float(np.mean(grid - baseline_grid)),
                grid_change_energy=float(np.sum(grid - baseline_grid)),
                grid_change_percent=(float(100 * (grid.sum() / baseline_grid.sum() - 1))
                                     if baseline_grid.sum() else None),
                grid_response_component_mean=float(np.mean(no_storage_grid - baseline_grid)),
                grid_storage_component_mean=float(np.mean(grid - no_storage_grid)))


def last_50_training_metrics(training):
    """Summarize the last update logged in each episode, not all optimizer updates."""
    rows = [row for row in training if 451 <= int(row["episode"]) <= 500]
    require([int(row["episode"]) for row in rows] == list(range(451, 501)),
            "Expected exactly 50 training records for episodes 451..500")
    result = {"episodes": [451, 500], "records": 50}
    for name in ("alpha", "entropy"):
        require(all(row[name] != "" for row in rows), f"Missing last-50 {name} metric")
        values = np.asarray([float(row[name]) for row in rows])
        require(np.isfinite(values).all(), f"Nonfinite last-50 {name} metric")
        if name == "alpha":
            require(np.all(values > 0), "Alpha must remain positive")
        result[name] = dict(mean=float(values.mean()), median=float(np.median(values)),
                            minimum=float(values.min()), maximum=float(values.max()),
                            last=float(values[-1]))
    return result


def analyze(run, low_hours, high_hours):
    v, profile, case = run["values"], run["profile"], run["case"]
    tou = np.asarray(profile["retail_price"], dtype=float)
    levels = np.unique(tou)
    require(len(levels) == 3, "Expected three original TOU levels")
    tou_means = {}
    for label, level in zip(("valley", "flat", "peak"), levels):
        hours = np.flatnonzero(tou == level)
        tou_means[label] = dict(tou_price=float(level), hours=hours.tolist(),
                                price_mean=float(v["price"][hours].mean()))
    weighted = lambda power: (float(np.dot(power, v["grid_price"]) / power.sum())
                              if power.sum() > 0 else None)
    storage = dict(charge_energy=float(v["charge"].sum()), discharge_energy=float(v["discharge"].sum()),
                   throughput=float((v["charge"] + v["discharge"]).sum()),
                   initial_soc=float(v["initial_soc"][0]), final_soc=float(v["soc"][-1]),
                   soc_change=float(v["soc"][-1] - v["initial_soc"][0]),
                   charge_weighted_wholesale=weighted(v["charge"]),
                   discharge_weighted_wholesale=weighted(v["discharge"]))
    storage["discharge_minus_charge_weighted_wholesale"] = (
        storage["discharge_weighted_wholesale"] - storage["charge_weighted_wholesale"]
        if storage["charge_weighted_wholesale"] is not None
        and storage["discharge_weighted_wholesale"] is not None else None)
    early_discharge = float(v["discharge"][EARLY_DISCHARGE_HOURS].sum())
    evening_discharge = float(v["discharge"][EVENING_DISCHARGE_HOURS].sum())
    total_discharge = storage["discharge_energy"]
    storage.update(discharge_h8_15_energy=early_discharge,
                   discharge_h8_15_percent=(100 * early_discharge / total_discharge
                                            if total_discharge > 0 else None),
                   discharge_h17_20_energy=evening_discharge,
                   discharge_h17_20_percent=(100 * evening_discharge / total_discharge
                                             if total_discharge > 0 else None))
    spec = run["env"]["storage"]
    for name, soc in (("h15_end", v["soc"][15]), ("h18_start", v["initial_soc"][18])):
        available = max(float(soc) - spec["soc_min"], 0.)
        storage[name] = dict(soc=float(soc), available_above_min=available,
                            potential_discharge_energy=available * spec["eta_dis"])
    return dict(sources=run["sources"], observation=run["cfg"]["observation"],
                training={**{k: run["complete"][k] for k in ("episodes", "steps", "updates", "wall_seconds")},
                          "last_50_logged_updates": last_50_training_metrics(run["training"])},
                profit=float(case["profit"]), storage_marginal_profit=float(case["storage_net"]),
                prices=dict(wholesale_pearson=pearson(v["price"], v["grid_price"]),
                            tou_pearson=pearson(v["price"], tou), tou_groups=tou_means,
                            tou_mean_ordered=bool(tou_means["valley"]["price_mean"]
                                                 <= tou_means["flat"]["price_mean"]
                                                 <= tou_means["peak"]["price_mean"]),
                            midday_mean=float(v["price"][PRICE_HOURS].mean()),
                            hour_22=float(v["price"][22]), hour_23=float(v["price"][23]),
                            change_22_to_23=float(v["price"][23] - v["price"][22])),
                curves={**{k: curve(v[k]) for k in ("baseline_load", "user_load", "grid_purchase")},
                        "baseline_grid": curve(np.maximum(v["baseline_load"] - v["wind"], 0)),
                        "same_price_zero_storage_grid": curve(np.maximum(v["user_load"] - v["wind"], 0))},
                storage=storage, midday_response=window_response(v, RESPONSE_HOURS),
                low_load_6h=window_response(v, low_hours), high_load_6h=window_response(v, high_hours),
                hourly_reduction_percent=(100 * (1 - v["user_load"] / v["baseline_load"])).tolist())


def numeric_deltas(before, after):
    """After minus before; percentage-valued metrics yield percentage-point deltas."""
    result = {}
    for key in before.keys() & after.keys():
        left, right = before[key], after[key]
        if isinstance(left, dict) and isinstance(right, dict):
            nested = numeric_deltas(left, right)
            if nested:
                result[key] = nested
        elif type(left) in (float, int) and type(right) in (float, int):
            result[key] = right - left
    return result


def main():
    runs = {label: read_run(path) for label, path in RUNS.items()}
    before, after = runs["before"], runs["after"]
    require(before["cfg"].keys() == after["cfg"].keys(), "Configuration keys differ")
    changed = sorted(k for k in before["cfg"] if before["cfg"][k] != after["cfg"][k])
    expected = {"observation": ("current6", "causal13")}
    require(changed == ["observation"], f"Only observation may differ: {changed}")
    for key, (old, new) in expected.items():
        require((before["cfg"][key], after["cfg"][key]) == (old, new),
                f"Unexpected {key}: expected {old} -> {new}")
    for key, value in (("warmup_steps", 1200), ("actor_lr", 2e-5), ("target_entropy", -2.0)):
        require(before["cfg"][key] == after["cfg"][key] == value,
                f"Both runs must retain {key}={value}")
    require(before["complete"]["updates"] == after["complete"]["updates"] == 86408,
            "Both runs must contain 86408 updates")
    for key in ("env", "generator", "profile"):
        require(before[key] == after[key], f"Effective {key} differs between runs")
    for key in ("baseline_load", "wind", "grid_price"):
        require(np.array_equal(before["values"][key], after["values"][key]),
                f"Realized exogenous {key} differs between runs")
    # Rank the unmodified baseline once; ties are resolved by ascending hour.
    baseline = before["values"]["baseline_load"]
    low_hours = sorted(sorted(range(24), key=lambda h: (baseline[h], h))[:6])
    high_hours = sorted(sorted(range(24), key=lambda h: (-baseline[h], h))[:6])
    results = {label: analyze(run, low_hours, high_hours) for label, run in runs.items()}
    output = dict(
        method="Arithmetic on two saved single-day trajectories; no policy/model/environment execution.",
        policy_loaded_or_evaluated=False,
        configuration_changes={key: {"before": old, "after": new}
                               for key, (old, new) in expected.items()},
        checks={"only_observation_differs": True, "effective_inputs_equal": True,
                "realized_exogenous_inputs_equal": True, "each_500_training_episodes": True,
                "each_test_n_1_hours_24": True, "noise_zero": True,
                "profit_reward_identity": True, "energy_balance": True, "frozen_source_sha256": True},
        definitions={
            "midday_price_hours": PRICE_HOURS, "midday_response_hours": RESPONSE_HOURS,
            "low_load_hours": low_hours, "high_load_hours": high_hours,
            "hour_selection": "Six smallest/largest original baseline loads; ties by ascending hour; actions never define labels.",
            "curve_rms_ramp": "sqrt(mean(diff(x)**2)), 23 within-day adjacent changes; no midnight wrap.",
            "curve_rms_level": "sqrt(mean(x**2)), distinct from ramp RMS; std uses population ddof=0.",
            "response_percent": "100*(1-sum(user_load)/sum(baseline_load)); positive means reduction.",
            "change_sign": "user/grid change metrics use actual-minus-baseline; positive means increase.",
            "baseline_grid": "max(original baseline_load-wind, 0), without response or storage.",
            "zero_storage": "max(recorded user_load-wind, 0), accounting counterfactual at the same recorded price.",
            "storage_prices": "Wholesale price weighted by actual charge/discharge power, not raw actions; null for no energy.",
            "discharge_h8_15_percent": "100*sum(actual discharge in h8..15)/sum(actual discharge over h0..23); null if no discharge.",
            "discharge_h17_20_energy": "sum(actual discharge in h17..20)*dt; current dt is explicitly checked to be one hour.",
            "available_soc": "h15_end uses soc[15] (16:00); h18_start uses initial_soc[18] (18:00). Available energy=max(SOC-soc_min,0).",
            "potential_discharge_energy": "available_above_min*eta_dis; does not account for power limits, time windows, or load absorption, and is not realized dispatch.",
            "last_50_training_metrics": "Mean/median/min/max/last over 50 episode-end records for episodes 451..500. Each record is that episode's last optimizer update, not its update average. Entropy is joint differential entropy in raw control coordinates and can be negative.",
            "update_count": "(12000-max(warmup_steps,batch_size,1)+1)*updates_per_step; 86408 for both runs.",
            "tou_groups": "Hours grouped by the three original profile retail_price levels, sorted valley/flat/peak.",
            "deltas": "after-before, same units; differences of percent metrics are percentage points, not relative percentages.",
            "units": "Recorded values unchanged. Study convention: price cents/kWh; power kW; energy kWh; profit cents.",
            "scope": "One paired fixed day and seed; not an independent generalization or convergence guarantee."},
        script=fingerprint(Path(__file__).resolve()), results=results,
        delta_after_minus_before=numeric_deltas(results["before"], results["after"]))
    # Fail rather than publish if any input changed during this offline computation.
    for run in runs.values():
        require(run["sources"] == [fingerprint(ROOT / s["path"]) for s in run["sources"]],
                f"{run['path']}: source files changed while reading")
    destination = RUNS["after"] / "offline_comparison.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for label, result in results.items():
        p, grid = result["prices"], result["curves"]["grid_purchase"]
        print(label, json.dumps(dict(observation=result["observation"], profit=result["profit"],
              wholesale_pearson=p["wholesale_pearson"], midday_price=p["midday_mean"],
              reduction_h11_15_pct=result["midday_response"]["reduction_percent"],
              price_h22_23_change=p["change_22_to_23"], grid_peak=grid["peak"],
              grid_rms_ramp=grid["rms_ramp"], grid_max_abs_jump=grid["max_abs_jump"],
              final_soc=result["storage"]["final_soc"],
              discharge_h8_15_percent=result["storage"]["discharge_h8_15_percent"],
              available_soc_h15_end=result["storage"]["h15_end"]["available_above_min"],
              available_soc_h18_start=result["storage"]["h18_start"]["available_above_min"],
              discharge_h17_20_energy=result["storage"]["discharge_h17_20_energy"],
              last50_alpha_mean=result["training"]["last_50_logged_updates"]["alpha"]["mean"],
              last50_entropy_mean=result["training"]["last_50_logged_updates"]["entropy"]["mean"]),
              ensure_ascii=False, allow_nan=False))
    print(f"All offline checks passed; wrote {destination}")


if __name__ == "__main__":
    main()
