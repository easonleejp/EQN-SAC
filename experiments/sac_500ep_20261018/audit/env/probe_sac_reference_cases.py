"""Calibrate the shape-cost scale and inspect price response on saved SAC cases.

Reads an existing evaluation and its config. No network is loaded or trained.
The optional price scan is a common-randomness counterfactual diagnostic of the
realized one-hour objective, not an executable policy with advance noise access.
"""

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from eqnsac.config import Config
from eqnsac.environment import Environment
from eqnsac.scenarios import LEVELS
from eqnsac.single_source import sample_case


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "runs/sac_baseline_500/evaluation_00500.json")
    parser.add_argument("--price-scan", action="store_true")
    args = parser.parse_args()
    source = args.evaluation.resolve()
    evaluation = json.loads(source.read_text("utf-8"))
    cfg_values = json.loads((source.parent / "config.json").read_text("utf-8"))
    env = Environment()
    price_scale = float(np.mean(env.profile["grid_price"]))
    load_scale = float(env.norm["max_load"])
    target = float(env.profile["baseline_load"].mean() * (1 - env.generator.cfg.target_penetration))
    rows = []
    for case in evaluation["cases"]:
        reference, active = case["reference_hourly"], case["hourly"]
        wind = np.array([row["wind"] for row in active])
        reference_user = np.array([row["user_load"] for row in reference])
        grid = np.maximum(reference_user - wind, 0)
        penalty = float(price_scale * load_scale * np.sum(((grid - target) / load_scale) ** 2))
        profit = sum(row["profit"] for row in reference)
        reference_excess = float(np.maximum(wind - reference_user, 0).sum())
        user_delta = sum(row["user_load"] for row in active) - float(reference_user.sum())
        storage_delta = sum(row["charge"] - row["discharge"] for row in active)
        excess_delta = sum(row["excess_energy"] for row in active) - reference_excess
        grid_delta = sum(row["grid_purchase"] for row in active) - float(grid.sum())
        rows.append({
            "case": case["case"], "reference_profit": profit,
            "reference_shape_cost_weight_one": penalty,
            "weight_for_5pct_penalty": 0.05 * profit / penalty,
            "weight_for_10pct_penalty": 0.10 * profit / penalty,
            "policy_grid_energy_change": grid_delta,
            "price_response_user_energy_change": user_delta,
            "storage_charge_minus_discharge": storage_delta,
            "excess_energy_change": excess_delta,
            "balance_residual": grid_delta - user_delta - storage_delta - excess_delta,
        })
    mean_profit = float(np.mean([row["reference_profit"] for row in rows]))
    mean_penalty = float(np.mean([row["reference_shape_cost_weight_one"] for row in rows]))
    calibration = {
        "source_evaluation": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reference": "profile retail tariff by hour, storage off; matched per-case response RNG",
        "cases": len(rows), "shape_target": target, "price_scale": price_scale, "load_scale": load_scale,
        "mean_reference_profit": mean_profit,
        "mean_reference_shape_cost_weight_one": mean_penalty,
        "weight_matching_5pct_mean_reference_penalty": 0.05 * mean_profit / mean_penalty,
        "weight_matching_10pct_mean_reference_penalty": 0.10 * mean_profit / mean_penalty,
        "profit_floor_5pct": 0.95 * mean_profit,
        "profit_floor_10pct": 0.90 * mean_profit,
        "energy_decomposition_mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("policy_grid_energy_change", "price_response_user_energy_change",
                        "storage_charge_minus_discharge", "excess_energy_change", "balance_residual")
        },
        "note": "Penalty-scale calibration only; realized economic loss must be checked on paired independent cases. The saved input is a real pure-SAC 500ep evaluation, not historical EQN.",
        "per_case": rows,
    }
    output = Path(__file__).parent
    write_json(output / "shape_weight_calibration_reference_27.json", calibration)
    print(json.dumps({key: value for key, value in calibration.items() if key != "per_case"}, ensure_ascii=False, indent=2))
    if not args.price_scan:
        return

    cfg = Config(**{
        key: cfg_values[key]
        for key in ("scenario_mode", "response_rng", "reference_seed", "load_scale_bound",
                    "wind_scale_bound", "price_scale_bound")
    }).validate()
    combinations = list(itertools.product(LEVELS, repeat=3))
    scan_rows = []
    for saved_case in evaluation["cases"]:
        index = saved_case["case"]
        scenario = sample_case(env, cfg, saved_case["scenario_seed"], combinations[index % 27])
        env.reset(scenario, saved_case["response_seed"])
        for hour in range(24):
            choices = []
            for action in np.linspace(-1, 1, 121):
                trial = copy.deepcopy(env)
                _, _, _, info = trial.step([action, 0])
                choices.append({"action": float(action), "price": info["price"],
                                "profit": info["profit"], "user_load": info["user_load"]})
            best = max(choices, key=lambda row: row["profit"])
            reference_row = saved_case["reference_hourly"][hour]
            scan_rows.append({"case": index, "hour": hour, **best,
                              "reference_price": reference_row["price"],
                              "reference_profit": reference_row["profit"],
                              "reference_user_load": reference_row["user_load"],
                              "min_price_profit": choices[0]["profit"],
                              "max_price_profit": choices[-1]["profit"]})
            env.step([best["action"], 0])
    by_hour = []
    for hour in range(24):
        subset = [row for row in scan_rows if row["hour"] == hour]
        by_hour.append({
            "hour": hour,
            "mean_best_price": float(np.mean([row["price"] for row in subset])),
            "min_best_price": min(row["price"] for row in subset),
            "max_best_price": max(row["price"] for row in subset),
            "upper_boundary_fraction": float(np.mean([row["action"] >= 1 - 1e-12 for row in subset])),
            "lower_boundary_fraction": float(np.mean([row["action"] <= -1 + 1e-12 for row in subset])),
        })
    result = {
        "source_evaluation": str(source.relative_to(ROOT)),
        "cases": len(evaluation["cases"]), "grid_points_per_hour": 121,
        "hours_scanned": len(scan_rows), "counterfactual_hourly_steps": len(scan_rows) * 121,
        "storage_policy": "off", "no_training_performed": True,
        "upper_boundary_fraction": float(np.mean([row["action"] >= 1 - 1e-12 for row in scan_rows])),
        "lower_boundary_fraction": float(np.mean([row["action"] <= -1 + 1e-12 for row in scan_rows])),
        "mean_daily_profit_at_scanned_price": float(sum(row["profit"] for row in scan_rows) / len(evaluation["cases"])),
        "mean_daily_user_energy_change_vs_reference": float(sum(row["user_load"] - row["reference_user_load"] for row in scan_rows) / len(evaluation["cases"])),
        "limitation": "This fixes each response RNG realization across action alternatives; it is a response-model diagnostic, not proof of the optimal partially observed stochastic policy or a learned SAC result.",
        "by_hour": by_hour, "per_case_hour": scan_rows,
    }
    write_json(output / "price_response_scan_27.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("per_case_hour", "by_hour")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
