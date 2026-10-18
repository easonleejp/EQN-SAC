"""Collect parameter evidence and boundary diagnostics without modifying inputs."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from response_parameter_probe import Environment,draws,response_grid,summarize,OUT


def main():
    scan=json.loads((OUT/'response_parameter_probe.json').read_text())
    elastic=json.loads((OUT/'exact_response_elasticity.json').read_text())
    scenarios=json.loads((OUT/'response_scenario_probe.json').read_text())
    env=Environment()
    case=env.generator.sample(20260914,'low','low','low')
    case.update(baseline_load=env.profile['baseline_load'],grid_price=env.profile['grid_price'])
    p=np.linspace(26.5,32.5,121);z=draws(256)
    ref=np.asarray(env.spec['hourly_demand_reference'])
    weak=response_grid(env,case,p,z,.02,ref)
    demand=weak['user'].mean(1,dtype=np.float64)
    derivative=np.gradient(demand,p,axis=1)
    e=derivative*p[None,:]/demand
    profit=weak['profit'].mean(1,dtype=np.float64)
    tou=env.profile['retail_price']
    aligned=response_grid(env,case,p,z,.001,tou)
    delta=(aligned['user'].mean(1,dtype=np.float64)/case['baseline_load'][:,None]-1)*100
    original=next(r for r in scan['variants'] if r['parameters']['name']=='original_k0.001')
    counts=[r['parameters']['name'] for r in scan['variants'] if 0<=r['low_load_mean_change_pct']<=5 and -5<=r['high_load_mean_change_pct']<=0]
    result={
        'scope':'仅诊断；未改主源码、原始数据、利润奖励或任何策略。全部参数变化仅内存反事实。',
        'current_original_parameter':dict(curvature_mean=.001,curvature_std=.0000828,demand_noise_std=50,
                                        rational_demand_derivative='approximately -1/(2*kappa) = -500 before participation; actual hourly local slope about -270 to -308',
                                        exact_actual_elasticity=elastic['results'][0]),
        'two_price_anchors':{
            'reference':'hourly_demand_reference 28.3–30.6: rational demand zero-response anchor, absent demand noise',
            'billing':'profile retail_price 26.5/28.5/32.5: baseline bill/relative-price-change anchor in satisfaction/participation; not the policy target',
            'current_tou_action_itself_responds':{'mean_low_tou_load_change_pct':scan['baseline_tou_low_mean_load_change_pct'],'mean_peak_tou_load_change_pct':scan['baseline_tou_high_mean_load_change_pct']},
            'aligned_reference_warning':'If reference is set to TOU, valley reference equals action minimum (cannot lower price for deterministic demand increase), peak reference equals action maximum (cannot raise price for deterministic demand reduction). Random demand noise may still cause tiny nonzero realizations.',
            'aligned_reference_at_valley_min_price_expected_pct_range':[float(delta[tou==26.5,0].min()),float(delta[tou==26.5,0].max())],
            'aligned_reference_at_peak_max_price_expected_pct_range':[float(delta[tou==32.5,-1].min()),float(delta[tou==32.5,-1].max())],
        },
        'original_economic_trend':{k:v for k,v in original.items() if k not in ['parameters','by_hour']},
        'nominal_tou_wholesale_order_conflicts':{'pairs':scan['tou_wholesale_pair_order_conflicts'],
            'interpretation':'Only disproves simultaneous strict hour-by-hour monotonic ordering; does not disprove overall low/flat/high trend requested by user.'},
        'weak_elasticity_counterfactual':{
            'curvature_mean':.02,'actual_elasticity_over_all_24x121_expected_demand_grid':[float(e.min()),float(e.max())],
            'minimum_profit_increment_per_price_grid_step':float(np.diff(profit,axis=1).min()),
            'maximum_price_optimal_hours':int((profit.argmax(1)==120).sum()),
            'economic_reason':'When grid import is positive, dprofit/dp = D+(p-c)*D_prime = D*[1+epsilon*(1-c/p)]. With these positive wholesale prices and weak negative own elasticity near -0.1/-0.2, this is positive. With zero grid import, derivative is D*(1+epsilon), also positive for weak elasticity. Thus the upper price bound is economically rational, not a SAC bug.',
            'paper_calibration':'Asadinejad & Tomsovic (2017), EPSR DOI 10.1016/j.epsr.2016.12.012, p220 Table2 assumes self=-0.1 and cross=+0.07 for their WECC case. This is not a universal range. -0.2 here is only a sensitivity benchmark.',
        },
        'finite_scan_reachability':{'parameter_combinations':len(scan['variants']),'definition':'diagnostic only: low-load quartile mean increase in [0,5]%, high-load quartile mean decrease in [-5,0]%; not a user-mandated threshold or added training constraint',
            'combinations_meeting_both_means':counts,'limitation':'No match in this finite parameter grid is not proof of global impossibility; hourly ranges also exceed mean ranges.'},
        'recommendations':[
            'Prioritize original-profit training/observation/hyperparameter experiments with original response parameters, as parent is doing; original expected economic optimum already has an overall wholesale/TOU trend.',
            'Treat curvature 0.0009, 0.001, 0.0011 as sensitivity probes if preserving current synthetic high-elasticity model. 0.0011 reduces mean valley increase, but does not solve excessive peak reduction and is not empirical calibration.',
            'If independently measured weak own-elasticity around -0.1 is the objective, curvature about 0.02 gives actual local elasticity [-0.188,-0.093] under current profiles. Expect all-day maximum retail pricing under the unchanged profit objective; do not label this failed learning.',
            'Do not align demand reference to TOU extremes, and do not fit each hourly reference to a desired price/dispatch trace. Uniform offset/amplitude sweeps quantify sensitivity only and lack customer-data validation.',
            'Global offset +1.5 with curvature 0.00125 gives nominal low/high-load mean changes about +3.13%/-6.40%, but 54.17% of hours use the price cap and hourly extremes reach +10.19%/-14.56%; this illustrates a trade-off, not a recommended calibrated replacement.',
            'PBDR still changes day-total demand; no delayed-load conservation or rebound exists. A small mean response does not guarantee small individual realizations or temporal shifting.',
        ],
        'scenario_robustness':scenarios['parameter_summaries'],
        'source_hashes':scan['source_hashes'],
    }
    (OUT/'response_diagnostic_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'weak_grid':result['weak_elasticity_counterfactual'],'aligned_reference':result['two_price_anchors'],'finite_scan':result['finite_scan_reachability']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
