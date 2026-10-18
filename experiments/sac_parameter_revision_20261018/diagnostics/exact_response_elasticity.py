"""Finite-difference elasticity using the actual unchanged response method.

Each lower/mid/upper price uses a restored identical RNG state. Prices at the
action bounds use one-sided differences. This is calibration diagnostics only.
"""
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from eqnsac.environment import Environment


def main():
    env=Environment()
    case=env.generator.sample(20260914,'low','low','low')
    case.update(baseline_load=env.profile['baseline_load'],grid_price=env.profile['grid_price'])
    reference=np.asarray(env.spec['hourly_demand_reference'])
    results=[]
    for k in [.001,.005,.01,.015,.02,.025]:
        env.demand.curvature_mean=k
        env.demand.curvature_std=.0828*k
        elasticities=np.empty((24,128,2))
        slopes=np.empty_like(elasticities)
        for draw in range(128):
            env.reset(case,940300000+draw)
            for hour in range(24):
                before=copy.deepcopy(env.rng.bit_generator.state)
                for anchor_index,anchor in enumerate([reference[hour],float(env.profile['retail_price'][hour])]):
                    low=max(anchor-.01,26.5);high=min(anchor+.01,32.5)
                    values=[]
                    for price in [low,anchor,high]:
                        env.rng.bit_generator.state=copy.deepcopy(before)
                        response,classes,_=env.response(np.full(3,price,dtype=np.float32))
                        values.append(float((classes-response).sum()))
                    derivative=(values[2]-values[0])/(high-low)
                    slopes[hour,draw,anchor_index]=derivative
                    elasticities[hour,draw,anchor_index]=derivative*anchor/values[1]
                env.t+=1
        hourly=[]
        for h in range(24):
            hourly.append(dict(hour=h,reference=float(reference[h]),billing_tou=float(env.profile['retail_price'][h]),
                               baseline_load=float(env.profile['baseline_load'][h]),
                               reference_elasticity_mean=float(elasticities[h,:,0].mean()),
                               billing_elasticity_mean=float(elasticities[h,:,1].mean()),
                               reference_demand_derivative_mean=float(slopes[h,:,0].mean()),
                               reference_elasticity_sd=float(elasticities[h,:,0].std())))
        results.append(dict(curvature_mean=k,curvature_std=.0828*k,
                            reference_mean_elasticity_range=[min(r['reference_elasticity_mean'] for r in hourly),max(r['reference_elasticity_mean'] for r in hourly)],
                            billing_mean_elasticity_range=[min(r['billing_elasticity_mean'] for r in hourly),max(r['billing_elasticity_mean'] for r in hourly)],
                            hourly=hourly))
    result=dict(equation_unchanged=True,noise_draws=128,seed_start=940300000,delta_price=.01,
                formula='epsilon = d(actual demand)/d(price) * anchor_price/actual_demand; average sample elasticity',
                common_random_numbers=True,results=results,
                calibration_caution='Elasticities -0.1/-0.2 are comparison targets supplied by parent task, not independently established universal customer constants. Constant absolute-demand slope generates different elasticity across hours/classes.')
    output=Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='hourly'} for r in results],indent=2))


if __name__=='__main__':main()
