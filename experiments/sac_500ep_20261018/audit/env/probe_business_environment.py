"""Read-only business-environment probes, independent of any trained SAC policy.

The linear programs below are perfect-foresight diagnostic comparators only.
They never replace actions during training and are not reported as SAC results.
"""

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from eqnsac.config import Config
from eqnsac.environment import Environment, FEATURES, FEATURE_INDEX, PRICE_INDICES, STORAGE_INDICES
from eqnsac.networks import storage_transform
from eqnsac.single_source import sample_case


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows, case, initial_soc):
    grid = np.array([r["grid_purchase"] for r in rows])
    load = np.array([r["user_load"] for r in rows])
    return {
        "profit": sum(r["profit"] for r in rows),
        "storage_net": sum(r["storage_marginal_profit_reward"] for r in rows),
        "storage_training_reward": sum(r["storage_training_reward"] for r in rows),
        "baseline_load_total": float(np.sum(case["baseline_load"])),
        "user_load_total": float(load.sum()),
        "user_load_peak": float(load.max()),
        "grid_purchase_total": float(grid.sum()),
        "grid_purchase_peak": float(grid.max()),
        "grid_purchase_valley": float(grid.min()),
        "grid_purchase_peak_valley": float(np.ptp(grid)),
        "grid_purchase_std": float(grid.std()),
        "grid_purchase_max_ramp": float(np.max(np.abs(np.diff(grid)))),
        "charge_total": sum(r["charge"] for r in rows),
        "discharge_total": sum(r["discharge"] for r in rows),
        "initial_soc": initial_soc,
        "final_soc": rows[-1]["soc"],
        "mean_satisfaction": sum(r["mean_satisfaction"] for r in rows) / 24,
    }


def rollout(case, price_actions, storage_actions=None):
    env = Environment()
    state = env.reset(case, 42)
    rows = []
    storage_actions = np.zeros(24) if storage_actions is None else storage_actions
    for hour in range(24):
        _, reward, done, info = env.step([price_actions[hour], storage_actions[hour]])
        grid = max(info["user_load"] + info["charge"] - info["discharge"] - float(env.wind[hour]), 0)
        info["grid_purchase"] = grid
        info["grid_price"] = float(env.price[hour])
        info["wind"] = float(env.wind[hour])
        info["reward_vector"] = reward.tolist()
        rows.append(info)
    return {"summary": summary(rows, case, env.storage.soc_init), "rows": rows}


def storage_lp(case, user_load, peak_first=False, peak_penalty=None):
    """Optimize charge/discharge with exact energy bounds and no export payment."""
    env = Environment()
    battery = env.storage
    horizon = 24
    include_peak = peak_first or peak_penalty is not None
    size = 97 if include_peak else 96
    objective = np.zeros(size)
    objective[:24] = battery.c_es
    objective[24:48] = battery.c_es
    objective[48:72] = np.asarray(case["grid_price"], dtype=np.float32)
    if peak_penalty is not None:
        objective[-1] = peak_penalty
    equality = np.zeros((24, size))
    equality_rhs = np.zeros(24)
    for hour in range(24):
        equality[hour, hour] = -battery.eta_ch
        equality[hour, 24 + hour] = 1 / battery.eta_dis
        equality[hour, 72 + hour] = 1
        if hour:
            equality[hour, 72 + hour - 1] = -1
        else:
            equality_rhs[hour] = battery.soc_init
    inequalities = np.zeros((48 if include_peak else 24, size))
    inequality_rhs = np.zeros(48 if include_peak else 24)
    net = np.asarray(user_load) - np.asarray(case["wind_power"], dtype=np.float32)
    for hour in range(24):
        inequalities[hour, hour] = 1
        inequalities[hour, 24 + hour] = -1
        inequalities[hour, 48 + hour] = -1
        inequality_rhs[hour] = -net[hour]
        if include_peak:
            inequalities[24 + hour, 48 + hour] = 1
            inequalities[24 + hour, 96] = -1
    bounds = (
        [(0, battery.p_ch_max)] * 24
        + [(0, battery.p_dis_max)] * 24
        + [(0, None)] * 24
        + [(battery.soc_min, battery.soc_max)] * 24
    )
    optimal_peak = None
    if include_peak:
        bounds.append((0, None))
    if peak_first:
        peak_objective = np.zeros(size)
        peak_objective[-1] = 1
        peak_result = linprog(peak_objective, A_ub=inequalities, b_ub=inequality_rhs,
                              A_eq=equality, b_eq=equality_rhs, bounds=bounds, method="highs")
        assert peak_result.success, peak_result.message
        optimal_peak = float(peak_result.x[-1])
        bounds[-1] = (optimal_peak, optimal_peak + 1e-6)
    result = linprog(objective, A_ub=inequalities, b_ub=inequality_rhs,
                     A_eq=equality, b_eq=equality_rhs, bounds=bounds, method="highs")
    assert result.success, result.message
    charge, discharge = result.x[:24], result.x[24:48]
    assert float(np.max(np.minimum(charge, discharge))) < 1e-6
    return {
        "actions": (charge / battery.p_ch_max - discharge / battery.p_dis_max).tolist(),
        "objective_cost": float(result.fun),
        "peak_penalty_coefficient": peak_penalty,
        "peak_first_optimum": optimal_peak,
        "max_simultaneous_charge_discharge": float(np.max(np.minimum(charge, discharge))),
    }


def myopic_price_probe(case):
    env = Environment()
    env.reset(case, 42)
    actions = []
    rows = []
    for hour in range(24):
        choices = []
        for action in np.linspace(-1, 1, 121):
            trial = copy.deepcopy(env)
            _, _, _, info = trial.step([action, 0])
            choices.append((info["profit"], float(action), info["price"], info["user_load"]))
        best = max(choices)
        actions.append(best[1])
        rows.append({"hour": hour, "best_price_action": best[1], "best_price": best[2],
                     "best_profit": best[0], "best_user_load": best[3],
                     "min_price_profit": choices[0][0], "max_price_profit": choices[-1][0]})
        env.step([best[1], 0])
    return {"actions": actions, "hourly_scan": rows, "rollout": rollout(case, actions)}


def alias_and_memory_probes(case):
    first = copy.deepcopy(case)
    second = copy.deepcopy(case)
    # An accepted synthetic pair: identical history/current net load, different gross load/wind.
    first["baseline_load"][12] = 2000
    first["wind_power"][12] = 0
    second["baseline_load"][12] = 3500
    second["wind_power"][12] = 1500
    environments = [Environment(), Environment()]
    states = []
    for env, scenario in zip(environments, (first, second)):
        env.reset(scenario, 42)
        for _ in range(12):
            env.step([0, 0])
        states.append(env.state())
    outcomes = []
    price_optima = []
    for env in environments:
        choices = []
        for action in np.linspace(-1, 1, 121):
            trial = copy.deepcopy(env)
            _, _, _, info = trial.step([action, 0])
            choices.append((info["profit"], float(action), info["price"]))
        best = max(choices)
        price_optima.append({"profit": best[0], "action": best[1], "price": best[2]})
        _, _, _, info = env.step([1, 0])
        outcomes.append({"gross_load": float(env.load[12]), "wind": float(env.wind[12]),
                         "user_load": info["user_load"], "response": info["total_response"],
                         "grid_purchase": max(info["user_load"] - float(env.wind[12]), 0),
                         "profit": info["profit"]})

    env = Environment()
    env.reset(case, 42)
    for _ in range(12):
        env.step([0.2, 0])
    changed_history = copy.deepcopy(env)
    changed_history.previous_price[:] = [27, 29, 31]
    changed_history.previous_response = -100000
    old_state, new_state = env.state(), changed_history.state()
    old_result, new_result = env.step([0.1, 0.2]), changed_history.step([0.1, 0.2])
    assert old_result[3] == new_result[3]
    np.testing.assert_array_equal(old_result[0], new_result[0])
    np.testing.assert_array_equal(old_result[1], new_result[1])

    gross_errors = []
    forecasts_errors = []
    previous_price_differences = []
    probe_env = Environment()
    for seed in range(20):
        scenario = probe_env.generator.sample(seed, "high", "high", "high")
        state = probe_env.reset(scenario, 42)
        for hour in range(24):
            gross = (float(state[FEATURE_INDEX["net_load"]]) * probe_env.norm["max_load"]
                     + float(state[FEATURE_INDEX["wind"]]) * probe_env.norm["max_pv"])
            gross_errors.append(abs(gross - float(probe_env.load[hour])))
            previous_price_differences.append(float(np.ptp(state[6:9])))
            for step in (1, 2, 3):
                forecast = probe_env.generator.load[min(23, hour + step)] * (
                    gross / probe_env.generator.load[hour]
                ) ** (probe_env.generator.cfg.load_rho**step)
                forecasts_errors.append(abs(forecast / probe_env.norm["max_load"] - state[12 + step]))
            state, _, _, _ = probe_env.step([np.sin(hour), 0])
    return {
        "synthetic_pair": {
            "valid_input_but_not_claimed_drawn_from_generator": True,
            "price4_exactly_equal": bool(np.array_equal(states[0][list(PRICE_INDICES)], states[1][list(PRICE_INDICES)])),
            "storage_compact10_exactly_equal": bool(np.array_equal(states[0][list(STORAGE_INDICES["compact10"])], states[1][list(STORAGE_INDICES["compact10"])])),
            "full19_exactly_equal": bool(np.array_equal(states[0], states[1])),
            "same_action_outcomes": outcomes,
            "price_scan_grid_points": 121,
            "same_observation_price_optima": price_optima,
        },
        "memory_mutation": {
            "pre_step_state_changed": not bool(np.array_equal(old_state, new_state)),
            "same_next_state_reward_done_and_info": True,
            "explanation": "Changing only previous_price/previous_response does not affect response or dynamics for the same new action and RNG state.",
        },
        "full19_redundancies": {
            "scenarios_checked": 20,
            "hours_checked": 480,
            "gross_load_reconstruction_max_absolute_error": max(gross_errors),
            "forecast_reconstruction_max_normalized_absolute_error": float(max(forecasts_errors)),
            "three_previous_price_slots_max_difference": max(previous_price_differences),
            "temperature_is_fixed_lookup_of_hour": True,
            "note": "Gross load and forecasts are recoverable from current net_load, wind and hour up to float32 rounding. Price/wind history can still convey extra information in three-source stochastic cases.",
        },
    }


def main():
    source_paths = sorted((ROOT / "eqnsac").glob("*.py")) + sorted((ROOT / "data").glob("*.json"))
    before = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    env = Environment()
    config = Config(scenario_mode="load_scale", load_scale_bound=0).validate()
    case = sample_case(env, config, 918)
    policies = {
        "constant_min_price_storage_off": rollout(case, np.full(24, -1.0)),
        "constant_mid_price_storage_off": rollout(case, np.zeros(24)),
        "constant_max_price_storage_off": rollout(case, np.full(24, 1.0)),
        "profile_retail_price_storage_off": rollout(case, (env.profile["retail_price"] - 29.5) / 3),
    }
    midpoint = policies["constant_mid_price_storage_off"]
    user_load = [row["user_load"] for row in midpoint["rows"]]
    economic = storage_lp(case, user_load)
    peak = storage_lp(case, user_load, peak_first=True)
    policies["offline_cost_optimal_storage_fixed_mid_price"] = rollout(case, np.zeros(24), economic["actions"])
    policies["offline_peak_first_then_cost_storage_fixed_mid_price"] = rollout(case, np.zeros(24), peak["actions"])
    target = float(env.generator.load.mean() * (1 - 0.2))
    mean_nominal_grid_price = float(env.generator.price.mean())
    load_normalizer = float(env.norm["max_load"])
    desired_power = target - (np.asarray(user_load) - np.asarray(case["wind_power"], dtype=np.float32))
    target_actions = np.where(desired_power >= 0, desired_power / env.storage.p_ch_max,
                              desired_power / env.storage.p_dis_max)
    policies["diagnostic_target_following_fixed_mid_price"] = rollout(case, np.zeros(24), target_actions)
    peak_tradeoff = []
    for coefficient in (2.0, 5.0, 10.0, 20.0):
        solution = storage_lp(case, user_load, peak_penalty=coefficient)
        policy = rollout(case, np.zeros(24), solution["actions"])
        peak_tradeoff.append({"coefficient": coefficient, "summary": policy["summary"],
                              "actions": solution["actions"]})

    pulse = np.zeros(24)
    pulse[18] = 1
    pulse_result = rollout(case, pulse)
    differences = [b["user_load"] - a["user_load"] for a, b in zip(midpoint["rows"], pulse_result["rows"])]
    no_rebound = {"price_changed_only_at_hour": 18, "hourly_user_load_differences": differences,
                  "day_energy_difference": float(sum(differences)),
                  "later_hours_max_absolute_difference": max(abs(d) for d in differences[19:])}

    battery = env.storage
    initial_state = env.reset(case)
    raw_zero, _ = storage_transform(torch.tensor(initial_state)[None], torch.zeros(1, 1), battery)
    final_charge = np.zeros(24)
    final_charge[-1] = 1
    last_charge = rollout(case, np.zeros(24), final_charge)
    capacity = {
        "soc_min": battery.soc_min, "soc_max": battery.soc_max, "soc_init": battery.soc_init,
        "initial_usable_energy": battery.soc_init - battery.soc_min,
        "usable_capacity": battery.soc_max - battery.soc_min,
        "max_charge_power": battery.p_ch_max, "max_discharge_power": battery.p_dis_max,
        "usable_capacity_over_rated_power_hours": (battery.soc_max - battery.soc_min) / battery.p_ch_max,
        "full_discharge_deliverable_energy": (battery.soc_max - battery.soc_min) * battery.eta_dis,
        "full_charge_grid_energy": (battery.soc_max - battery.soc_min) / battery.eta_ch,
        "round_trip_efficiency": battery.eta_ch * battery.eta_dis,
        "max_charge_power_over_nominal_peak_load": battery.p_ch_max / float(env.generator.load.max()),
        "actor_raw_zero_at_initial_soc_physical_action": float(raw_zero.item()),
        "actor_raw_zero_at_initial_soc_charge_power": float(raw_zero.item()) * battery.p_ch_max,
        "last_hour_full_charge_profit_difference_vs_off": last_charge["summary"]["profit"] - midpoint["summary"]["profit"],
        "last_hour_full_charge_final_usable_energy": last_charge["summary"]["final_soc"] - battery.soc_min,
        "daily_reset_discards_final_usable_energy_and_does_not_gift_initial_usable_energy": True,
    }
    for name in ("offline_cost_optimal_storage_fixed_mid_price", "offline_peak_first_then_cost_storage_fixed_mid_price"):
        result = policies[name]["summary"]
        result["energy_balance_residual"] = (result["final_soc"] - result["initial_soc"]
            - battery.eta_ch * result["charge_total"] + result["discharge_total"] / battery.eta_dis)
        result["true_storage_benefit_vs_off"] = result["profit"] - midpoint["summary"]["profit"]
        result["storage_training_reward_increment_vs_off"] = result["storage_training_reward"] - midpoint["summary"]["storage_training_reward"]

    rewards = np.array([r["profit"] for r in midpoint["rows"]])
    result = {
        "scope": "Untrained shared business-environment diagnostic; not a pure SAC or EQN training result.",
        "case": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in case.items()},
        "response_seed": 42,
        "units": "Original exported numerical units; currency and price scaling not independently verified.",
        "capacity_and_terminal": capacity,
        "state_alias_and_redundancy": alias_and_memory_probes(case),
        "price_pulse_no_rebound": no_rebound,
        "policies": policies,
        "myopic_price_scan": myopic_price_probe(case),
        "linear_program_diagnostics": {"cost_optimal": economic, "peak_first": peak},
        "peak_penalty_tradeoff": {
            "objective": "Operating cost + coefficient * daily maximum grid purchase; offline fixed-price diagnostic only.",
            "coefficient_units": "Original unverified monetary units per unit of daily peak power; not a calibrated tariff recommendation.",
            "results": peak_tradeoff,
        },
        "profit_reward_scale": {
            "hourly_min": float(rewards.min()), "hourly_mean": float(rewards.mean()), "hourly_max": float(rewards.max()),
            "daily_sum": float(rewards.sum()), "configured_scale": Config().reward_scale,
            "scaled_hourly_mean": float(rewards.mean() * Config().reward_scale),
            "scaled_daily_sum": float(rewards.sum() * Config().reward_scale),
            "storage_off_mixed_reward": midpoint["summary"]["storage_training_reward"],
            "storage_off_true_marginal_reward": midpoint["summary"]["storage_net"],
        },
        "separable_candidate_changes": [
            {
                "name": "compact11_observation_only",
                "features": ["price", "net_load", "wind", "soc", "hour",
                             "price_lag1", "price_lag2", "price_lag3",
                             "wind_lag1", "wind_lag2", "wind_lag3"],
                "scope": "Controlled follow-up to the unchanged full19 SAC baseline; leave reward and physical parameters unchanged.",
                "rationale": "Fixed hourly temperature, three equal previous-price slots, unused previous response and three algebraically recoverable forecasts are redundant under the current environment assumptions. Explicit forecast features may still help optimization, so this is an ablation candidate rather than a proven improvement.",
            },
            {
                "name": "explicit_peak_objective_if_peak_shaving_is_required",
                "proposal": "Augment observations with running peak; apply penalty coefficient * (new_running_peak - old_running_peak). The sum equals coefficient * daily peak when initialized at zero.",
                "coefficient_diagnostic_values": [2.0, 5.0, 10.0, 20.0],
                "scope": "Independent objective change, not part of a pure-profit baseline. Calibrate the unit and operational tradeoff before selecting a coefficient.",
            },
            {
                "name": "separate_price_curtailment_from_energy_shifting",
                "proposal": "Current demand response is per-hour curtailment/increase with no backlog or rebound. If a deferrable-load interpretation is required, implement energy balance/backlog and deadline state in a separate environment version.",
            },
            {
                "name": "preserve_physical_data_and_clarify_terminal_scope",
                "proposal": "Do not change capacity, efficiency, initial SOC or price units to improve a 500-episode curve. Initial usable energy is zero. Free final SOC models a one-day problem; continuous operation would require explicit carry-over or a justified terminal value in another environment version.",
            },
        ],
        "source_sha256": before,
    }
    price_scan = result["myopic_price_scan"]
    price_scan["boundary_summary"] = {
        "at_upper_count": sum(a >= 1 - 1e-12 for a in price_scan["actions"]),
        "at_lower_count": sum(a <= -1 + 1e-12 for a in price_scan["actions"]),
        "total_hours": 24,
        "upper_hours": [i for i, a in enumerate(price_scan["actions"]) if a >= 1 - 1e-12],
        "note": "121-point deterministic one-hour profit scan with storage off; not a trained policy or proof for all scenarios.",
    }
    shaping_policies = dict(policies)
    shaping_policies["myopic_price_storage_off"] = price_scan["rollout"]
    shape_rows = {}
    for name, policy in shaping_policies.items():
        grid = np.array([row["grid_purchase"] for row in policy["rows"]])
        unit_penalty = float(mean_nominal_grid_price * load_normalizer
                             * np.sum(((grid - target) / load_normalizer) ** 2))
        shape_rows[name] = {
            "profit": policy["summary"]["profit"],
            "grid_mean": float(grid.mean()),
            "grid_variance": float(grid.var()),
            "squared_mean_bias": float((grid.mean() - target) ** 2),
            "unweighted_daily_shape_penalty": unit_penalty,
            "shaped_daily_objective": {str(w): policy["summary"]["profit"] - w * unit_penalty
                                       for w in (0.1, 0.25, 0.5, 1.0, 2.0)},
            "profit_difference_vs_fixed_mid_no_storage": policy["summary"]["profit"] - midpoint["summary"]["profit"],
            "final_usable_energy": policy["summary"]["final_soc"] - env.storage.soc_min,
        }
    reference_penalty = shape_rows["constant_mid_price_storage_off"]["unweighted_daily_shape_penalty"]
    result["proposed_fixed_target_shape_penalty"] = {
        "target": target,
        "mean_nominal_grid_price": mean_nominal_grid_price,
        "load_normalizer": load_normalizer,
        "formula": "weight * mean_nominal_grid_price * max_load * ((grid_purchase - target) / max_load)^2 per hour",
        "information_boundary": "Target uses known nominal profile and configured expected wind penetration; no future realized data is used.",
        "policies": shape_rows,
        "calibration_relative_to_fixed_mid_price_no_storage": {
            "reference_profit": midpoint["summary"]["profit"],
            "reference_shape_penalty_at_weight_one": reference_penalty,
            "five_percent_profit_budget": 0.05 * midpoint["summary"]["profit"],
            "ten_percent_profit_budget": 0.10 * midpoint["summary"]["profit"],
            "weight_matching_five_percent_reference_penalty": 0.05 * midpoint["summary"]["profit"] / reference_penalty,
            "weight_matching_ten_percent_reference_penalty": 0.10 * midpoint["summary"]["profit"] / reference_penalty,
            "warning": "Matching the penalty scale to a 5%/10% profit budget is an initialization heuristic, not a guarantee on realized profit loss. Validate actual paired economic performance on held-out seeds.",
        },
        "traps": [
            "The squared penalty equals variance plus squared mean bias around a fixed target. It rewards raising an economical low mean through demand increases or charging, not only shaving high peaks.",
            "The nominal target ignores price-induced demand changes; different price policies have different daily net energy. Wind reference realization also differs from expected 20% penetration.",
            "Unused terminal energy can reduce the shape penalty while losing cash. Audit final usable SOC and charging in the final hours separately.",
            "An L2 tracking penalty does not directly constrain the maximum grid purchase or ramps. Always report peak, valley, standard deviation and max ramp in addition to shaped reward.",
            "Coefficient units depend on original unverified price/power units. This is an operational preference regularizer, not an identified demand-charge tariff.",
        ],
    }
    after = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    result["source_unchanged_during_probe"] = before == after
    output = Path(__file__).parent / "business_environment_probe.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    print(json.dumps({
        "capacity_and_terminal": capacity,
        "state_alias_and_redundancy": result["state_alias_and_redundancy"],
        "price_pulse_no_rebound": no_rebound,
        "policy_summaries": {k: v["summary"] for k, v in policies.items()},
        "myopic_price_summary": result["myopic_price_scan"]["rollout"]["summary"],
        "myopic_price_actions": result["myopic_price_scan"]["actions"],
        "profit_reward_scale": result["profit_reward_scale"],
        "peak_penalty_tradeoff": [{"coefficient": r["coefficient"],
                                    "profit": r["summary"]["profit"],
                                    "grid_peak": r["summary"]["grid_purchase_peak"]}
                                   for r in peak_tradeoff],
        "source_unchanged_during_probe": result["source_unchanged_during_probe"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
