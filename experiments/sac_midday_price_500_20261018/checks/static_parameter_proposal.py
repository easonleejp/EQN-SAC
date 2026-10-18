"""Static quadratic-utility proposal; never loads an Actor or steps a policy.

Reads the previous run, scans its deterministic hourly response/profit equation,
and writes a parameter proposal. No active configuration or source is changed.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
module = importlib.util.spec_from_file_location(
    "static_response_probe", ROOT / "checks/sac_revision/response_parameter_probe.py")
probe = importlib.util.module_from_spec(module)
module.loader.exec_module(probe)


def main():
    source = ROOT / "runs/sac_hourly_utility_500/test_1ep.json"
    saved = json.loads(source.read_text())
    rows = saved["cases"][0]["hourly"]
    spec = copy.deepcopy(saved["metadata"]["environment"])
    assert all(spec["demand"][key] == 0 for key in
               ("curvature_std", "demand_noise_std", "participation_noise_std"))
    env = probe.Environment(spec=spec)
    case = {key: np.asarray([row[field] for row in rows]) for key, field in
            (("baseline_load", "baseline_load"), ("wind_power", "wind"),
             ("grid_price", "grid_price"))}
    hours = np.arange(24)
    load = case["baseline_load"]
    old_a = np.asarray(spec["hourly_utility"]["a"])
    old_b = np.asarray(spec["hourly_utility"]["b"])
    old_k = -old_a
    old_ref = old_b + 2 * old_a * load
    # Constant daytime utility curvature/reference with raised-cosine shoulders.
    weight = np.zeros(24)
    weight[9:16] = 1
    weight[[7, 8, 16, 17]] = [.25, .75, .75, .25]
    kappa = (1 - weight) * old_k + weight * .000825
    reference = (1 - weight) * old_ref + weight * 30.5
    a = -kappa
    b = reference + 2 * kappa * load
    # Preserve unchanged hours bit for bit instead of round-tripping b.
    a[weight == 0] = old_a[weight == 0]
    b[weight == 0] = old_b[weight == 0]
    prices = np.linspace(26.5, 32.5, 1201)
    zero_noise = np.zeros((24, 1, 11))
    def scan(k, ref):
        values = probe.response_grid(env, case, prices, zero_noise,
                                     k[:, None], ref, std_ratio=0)
        idx = values["profit"][:, 0].argmax(-1)
        best_prices = prices[idx]
        change = 100 * (values["user"][hours, 0, idx] / load - 1)
        cap_change = 100 * (values["user"][:, 0, -1] / load - 1)
        return values, idx, best_prices, change, cap_change
    previous = scan(old_k, old_ref)
    candidate = scan(kappa, reference)
    pv, pi, pp, pl, pc = previous
    cv, ci, cp, cl, cc = candidate
    # Only static response calls: no Environment.step(), policy, or storage path.
    proposal_spec = copy.deepcopy(spec)
    proposal_spec["hourly_utility"] = {"a": a.tolist(), "b": b.tolist()}
    proposal_spec["demand"]["hourly_curvature_mean"] = kappa.tolist()
    proposal_spec["hourly_demand_reference"] = reference.tolist()
    checked = probe.Environment(spec=proposal_spec)
    errors, old_price_load = [], []
    saved_prices = np.asarray([row["price"] for row in rows])
    for selected in (np.full(24, 26.5), np.full(24, 32.5), cp, saved_prices):
        checked.reset(case, 42)
        actual_load = []
        for hour in range(24):
            checked.t = hour
            price = float(selected[hour])
            response, classes, _ = checked.response(np.full(3, price, np.float32))
            actual = float((classes - response).sum())
            actual_load.append(actual)
            predicted = probe.response_grid(env, case, [price], zero_noise,
                                            kappa[:, None], reference, std_ratio=0)
            errors.append(abs(actual - float(predicted["user"][hour, 0, 0])))
        if np.array_equal(selected, saved_prices):
            old_price_load = np.asarray(actual_load)
    assert max(errors) < .003, max(errors)
    assert np.all(a < 0)
    assert np.array_equal(a[weight == 0], old_a[weight == 0])
    assert np.array_equal(b[weight == 0], old_b[weight == 0])
    output_rows = []
    curve_indices = [int(round((price - 26.5) / .005))
                     for price in (29.5, 30, 30.5, 31, 31.5, 32, 32.5)]
    shares = np.asarray(env.generator.shares)
    for hour in range(24):
        output_rows.append(dict(
            hour=hour, weight=float(weight[hour]), a=float(a[hour]), b=float(b[hour]),
            kappa=float(kappa[hour]), reference=float(reference[hour]),
            class_a=(a[hour] / shares[hour]).tolist(), class_b=[float(b[hour])] * 3,
            previous_saved_price=float(saved_prices[hour]),
            previous_saved_load_change_pct=float(100 * (rows[hour]["user_load"] / load[hour] - 1)),
            previous_static_best_price=float(pp[hour]),
            previous_static_best_load_change_pct=float(pl[hour]),
            previous_static_cap_profit_gap=float(pv['profit'][hour, 0, pi[hour]] - pv['profit'][hour, 0, -1]),
            candidate_static_best_price=float(cp[hour]),
            candidate_static_best_load_change_pct=float(cl[hour]),
            candidate_load_change_at_cap_pct=float(cc[hour]),
            candidate_load_change_at_previous_saved_price_pct=float(100 * (old_price_load[hour] / load[hour] - 1)),
            candidate_static_cap_profit_gap=float(cv['profit'][hour, 0, ci[hour]] - cv['profit'][hour, 0, -1]),
            curve_points=[dict(price=float(prices[j]), previous_profit=float(pv['profit'][hour, 0, j]),
                               candidate_profit=float(cv['profit'][hour, 0, j]),
                               candidate_load_change_pct=float(100 * (cv['user'][hour, 0, j] / load[hour] - 1)))
                          for j in curve_indices]))
    result = dict(
        status="static_parameter_proposal_only_not_trained", baseline_run="runs/sac_hourly_utility_500",
        scope="仅24对a/b改变；固定输入、设备、新能源、dispatch17、超参、奖励和响应公式均保持。无目标动作或奖励项。",
        proposal_name="midday_common_curvature_reference_with_cosine_shoulders",
        construction={"core_hours": list(range(9, 16)), "core_kappa": .000825, "core_reference": 30.5,
                      "shoulder_weights": {"7": .25, "8": .75, "16": .75, "17": .25},
                      "other_weights": 0,
                      "equations": ["kappa_new=(1-w)*kappa_old+w*0.000825", "reference_new=(1-w)*reference_old+w*30.5", "a_new=-kappa_new", "b_new=reference_new+2*kappa_new*L0"],
                      "shoulder_basis": "两侧3小时升余弦过渡：h6/7/8/9权重0/.25/.75/1，h15/16/17/18为1/.75/.25/0；未改变小时原样保留。"},
        algebra={"utility": "U_t(L)=a_t L^2+b_t L; per-class a_tj=a_t/share_tj, b_tj=b_t",
                 "runtime": "kappa=-a; reference=b+2*a*current_class_load_sum",
                 "local_explanation": "暂将参与系数mu视为常数，D=L0-mu*(p-reference)/(2*kappa)，则内点利润最优p*约为(c+reference)/2+kappa*L0/mu。减小kappa可降低利润最优价，但仅减reference会同时加重削减；联合选择a,b可兼顾价格和响应。实际诊断仍计算完整三次参与迭代，未用这个局部近似替代公式。",
                 "not_reduced_elasticity": "较小kappa意味着理性需求价格敏感性增加，不能称为简单减弱弹性；削减是否温和必须与reference及利润最优价一同核验。"},
        candidate={"hourly_utility_a": a.tolist(), "hourly_utility_b": b.tolist(),
                   "hourly_kappa": kappa.tolist(), "hourly_demand_reference": reference.tolist(),
                   "class_shares": shares.tolist(), "hourly": output_rows},
        summary={"price_window_hours": list(range(10, 15)), "response_window_hours": list(range(11, 16)),
                 "previous_saved_mean_price_10_14": float(saved_prices[10:15].mean()),
                 "previous_static_mean_price_10_14": float(pp[10:15].mean()),
                 "candidate_static_mean_price_10_14": float(cp[10:15].mean()),
                 "static_mean_price_change_10_14": float(cp[10:15].mean() - pp[10:15].mean()),
                 "candidate_static_prices_9_15": cp[9:16].tolist(),
                 "candidate_static_load_change_pct_11_15": cl[11:16].tolist(),
                 "candidate_max_reduction_pct_over_entire_price_grid_9_15": float(np.max(100 * (1 - cv['user'][9:16, 0] / load[9:16, None]))),
                 "previous_static_cap_hours": np.flatnonzero(pi == 1200).tolist(),
                 "candidate_static_cap_hours": np.flatnonzero(ci == 1200).tolist(),
                 "candidate_wholesale_price_correlation": float(np.corrcoef(cp, case["grid_price"])[0, 1]),
                 "candidate_mean_price_by_TOU": {str(tou): float(cp[env.profile["retail_price"] == tou].mean()) for tou in (26.5, 28.5, 32.5)}},
        static_qa={"grid": {"min": 26.5, "max": 32.5, "step": .005, "points": 1201},
                   "noise": "与既有固定日相同，所有响应噪声为0", "kernel_max_load_error": float(max(errors)),
                   "verified_reference_and_curvature": "候选在内存中配置到真实Environment.response；仅单小时静态调用，不执行Environment.step/Actor。",
                   "unchanged_hours_bitwise_preserved": True,
                   "class_arrays_each_length_three": all(len(row["class_a"]) == len(row["class_b"]) == 3 for row in output_rows)},
        interpretation=["上轮h9–12的静态最优已低于价格上限；Actor顶价并非都由经济目标解释。h13–15原静态目标确实偏好上限。",
                        "目标窗口是透明的合成情景选择，缺乏实测用户标定；这些静态最优价只用于离线核验，不能输入动作、奖励、训练标签或约束。",
                        "新参数令h9–15全部零售价网格下削减不超过约18.33%，可降低策略继续顶价时的强削减风险；这不是添加需求响应上限。",
                        "静态最优无储能，不能替代500轮SAC的学习结果或联合调度最优；已保存的价格只作为算术输入，没有加载旧Actor。",
                        "减轻削减可能增加购电负荷或影响削峰，本轮不改变利润目标以强求形状。"],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [source, ROOT / "eqnsac/environment.py", ROOT / "checks/sac_revision/response_parameter_probe.py", Path(__file__)]})
    out = Path(__file__).with_name("parameter_proposal.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"path": str(out), "summary": result["summary"], "kernel_error": max(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
