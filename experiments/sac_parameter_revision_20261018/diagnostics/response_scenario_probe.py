"""Independent 27-combination sensitivity check; no actor, storage or training."""
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from response_parameter_probe import Environment, draws, response_grid, summarize, OUT


def main():
    env=Environment()
    prices=np.linspace(26.5,32.5,121)
    reference=np.asarray(env.spec['hourly_demand_reference'])
    z=draws(64,940200000)
    grouped={str(k):[] for k in [.00075,.001,.0011]}
    for index,levels in enumerate(itertools.product(['low','medium','high'],repeat=3)):
        case=env.generator.sample(940100000+index,*levels)
        for k in [.00075,.001,.0011]:
            parameter=dict(curvature_mean=k,curvature_std=k*.0828,reference=reference.tolist())
            row=summarize(env,case,prices,response_grid(env,case,prices,z,k,reference),parameter)
            row.update(case=index,seed=940100000+index,levels=levels,
                       hours_grid_price_above_retail_cap=int((case['grid_price']>32.5).sum()))
            row['by_hour']=[{a:b for a,b in r.items() if not a.startswith('prices_with_')} for r in row['by_hour']]
            del row['parameters']
            grouped[str(k)].append(row)
    summaries={}
    for k,rows in grouped.items():
        keys=['daily_expected_profit','mean_optimal_price','upper_fraction','lower_fraction',
              'price_wholesale_pearson','price_wholesale_spearman','daily_user_energy_change_pct',
              'low_load_mean_change_pct','high_load_mean_change_pct','low_load_increase_0_to_5pct_hours',
              'high_load_decrease_0_to_5pct_hours','mean_price_tou_low','mean_price_tou_flat','mean_price_tou_high',
              'hours_grid_price_above_retail_cap']
        summaries[k]={key:float(np.mean([r[key] for r in rows if r[key] is not None])) for key in keys}
    result=dict(no_training=True,scenario_seed_start=940100000,cases=27,level_combinations_equally_weighted=True,
                response_draws_per_case=64,response_seed_start=940200000,price_grid_points=121,
                scope='Expected per-hour original profit, no storage. Unchanged formula, original hourly reference. Different curvature parameter only.',
                parameter_summaries=summaries,case_results=grouped,
                caution='Diagnostic scenario suite, not a heldout validation of a learned SAC model. No monetary units inferred.')
    (OUT/'response_scenario_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
