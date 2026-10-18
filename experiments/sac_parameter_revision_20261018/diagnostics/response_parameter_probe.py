"""Read-only parameter diagnostics for the unchanged hourly response formula.

All parameter changes are in-memory counterfactuals. No policy is trained and
no reward, constraints, data files or source modules are changed. The vectorized
kernel is checked against Environment.response before reporting its sweeps.
"""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from eqnsac.environment import Environment

OUT = Path(__file__).parent


def draws(count=256, seed=940000000):
    z = np.empty((24, count, 11))
    for sample in range(count):
        rng = np.random.default_rng(seed + sample)
        for hour in range(24):
            z[hour, sample, 0] = rng.normal()
            z[hour, sample, 1] = rng.normal()
            for iteration in range(3):
                z[hour, sample, 2+iteration*3:5+iteration*3] = rng.normal(size=3)
    return z


def response_grid(env, scenario, prices, z, curvature_mean, reference, std_ratio=.0828):
    # Axes: hour, response realization, shared retail price, user class.
    load = np.asarray(scenario['baseline_load'], np.float32)
    classes = load.astype(float)[:, None] * env.generator.shares
    classes[:, 2] = load.astype(float) - classes[:, :2].sum(1)
    classes = classes.astype(np.float32)[:, None, None, :]
    shares = classes / np.maximum(classes.sum(-1, keepdims=True), np.float32(1e-8))
    p = np.asarray(prices, np.float32)[None, None, :, None]
    k = np.maximum(curvature_mean + curvature_mean*std_ratio*z[..., 0], env.demand.min_curvature).astype(np.float32)[..., None, None]
    noise = (env.demand.demand_noise_std*z[..., 1]).astype(np.float32)[..., None, None]
    r = np.asarray(reference, np.float32)[:, None, None, None]
    rational = classes + (p-r) / (2*(-k/np.maximum(shares, np.float32(1e-8)))) + shares*noise
    theoretical = (classes-rational).astype(np.float32)
    billing = env.profile['retail_price'][:, None, None, None]
    change = abs(p-billing)/np.maximum(billing, np.float32(1e-8))
    temp = (20/(abs(env.profile['temperature']-22)+.001)).astype(np.float32)[:, None, None, None]
    actual = theoretical.copy()
    for iteration in range(3):
        comfort = 1-actual**2/np.maximum(classes**2, np.float32(1e-8))
        bill = p*np.maximum(classes-actual, 0)
        baseline_bill = billing*classes
        economic = 1-(bill-baseline_bill)/np.maximum(abs(baseline_bill), np.float32(1e-8))
        satisfaction = np.clip(env.user.comfort_weight*comfort+env.user.comp_weight*economic, 0, 5)
        participation = 1/(1+np.exp(-env.user.lambda_s*(satisfaction-env.user.s_ref)))
        factor = np.clip(.42/(1+np.exp(-4.2*(change-.05))) + .22/(1+np.exp(-(temp-1)))
                         + .36/(1+np.exp(-4*(participation-.5)))
                         + (.025*z[..., 2+iteration*3:5+iteration*3]).astype(np.float32)[:, :, None, :], 0, 1)
        actual = np.minimum(factor*theoretical, classes)
    user = (classes-actual).sum(-1)
    revenue = (p*(classes-actual)).sum(-1)
    grid = np.maximum(user-np.asarray(scenario['wind_power'], np.float32)[:, None, None], 0)
    profit = revenue-grid*np.asarray(scenario['grid_price'], np.float32)[:, None, None]
    return dict(user=user, profit=profit, response=actual.sum(-1), satisfaction=satisfaction.mean(-1))


def summarize(env, scenario, prices, values, parameter):
    mean = {k:v.mean(axis=1, dtype=np.float64) for k,v in values.items()}
    best_index = mean['profit'].argmax(axis=-1)
    h=np.arange(24)
    bestp = np.asarray(prices)[best_index]
    bestload = mean['user'][h,best_index]
    base = np.asarray(scenario['baseline_load'])
    delta = (bestload/base-1)*100
    low=base<=np.quantile(base,.25); high=base>=np.quantile(base,.75)
    tou=env.profile['retail_price']; wholesale=np.asarray(scenario['grid_price'])
    rows=[]
    for hour in h:
        desired = mean['user'][hour]/base[hour]-1
        rows.append(dict(hour=int(hour),baseline_load=float(base[hour]),grid_price=float(wholesale[hour]),
                         nominal_tou=float(tou[hour]), reference=float(parameter['reference'][hour]),
                         expected_profit_optimal_price=float(bestp[hour]),user_load=float(bestload[hour]),
                         load_change_pct=float(delta[hour]),optimal_profit=float(mean['profit'][hour,best_index[hour]]),
                         response_at_min_price_pct=float((mean['user'][hour,0]/base[hour]-1)*100),
                         response_at_max_price_pct=float((mean['user'][hour,-1]/base[hour]-1)*100),
                         prices_with_increase_0_to_5pct=np.asarray(prices)[(desired>=0)&(desired<=.05)].tolist(),
                         prices_with_decrease_0_to_5pct=np.asarray(prices)[(desired<=0)&(desired>=-.05)].tolist()))
    return dict(parameters=parameter, daily_expected_profit=float(mean['profit'][h,best_index].sum()),
                mean_optimal_price=float(bestp.mean()),upper_fraction=float(np.mean(best_index==len(prices)-1)),
                lower_fraction=float(np.mean(best_index==0)),price_wholesale_pearson=float(np.corrcoef(bestp,wholesale)[0,1]) if np.ptp(bestp)>0 else None,
                price_wholesale_spearman=float(spearmanr(bestp,wholesale).statistic) if np.ptp(bestp)>0 else None,
                daily_user_energy_change_pct=float((bestload.sum()/base.sum()-1)*100),
                low_load_hours=np.flatnonzero(low).tolist(),high_load_hours=np.flatnonzero(high).tolist(),
                low_load_mean_change_pct=float(delta[low].mean()),high_load_mean_change_pct=float(delta[high].mean()),
                maximum_increase_pct=float(delta.max()),maximum_decrease_pct=float(delta.min()),
                low_load_increase_0_to_5pct_hours=int(((delta[low]>=0)&(delta[low]<=5)).sum()),
                high_load_decrease_0_to_5pct_hours=int(((delta[high]<=0)&(delta[high]>=-5)).sum()),
                mean_price_tou_low=float(bestp[tou==26.5].mean()),mean_price_tou_flat=float(bestp[tou==28.5].mean()),
                mean_price_tou_high=float(bestp[tou==32.5].mean()),by_hour=rows)


def main():
    env=Environment()
    case=env.generator.sample(20260914,'low','low','low')
    case.update(baseline_load=env.profile['baseline_load'].copy(),grid_price=env.profile['grid_price'].copy())
    z=draws()
    prices=np.linspace(26.5,32.5,121)
    reference=np.asarray(env.spec['hourly_demand_reference'])
    # Behavioral guard: same RNG stream, all hours, three endpoint/midpoint prices.
    checked=response_grid(env,case,[26.5,29.5,32.5],z[:,:1],.001,reference)
    errors=[]
    for j,p in enumerate([26.5,29.5,32.5]):
        env.reset(case,940000000)
        for h in range(24):
            actual,classes,_=env.response(np.full(3,p,np.float32))
            errors.append(abs(float(actual.sum())-float(checked['response'][h,0,j])))
            env.t+=1
    if max(errors)>.003: raise AssertionError(('kernel mismatch',max(errors)))
    variants=[]
    for k in [.0005,.00075,.0009,.001,.0011,.00125,.0015,.002,.003,.005,.01,.015,.02,.025]:
        variants.append((f'original_k{k}',k,reference.copy(),'original hourly reference'))
    for amplitude in [0,.5,1.5,2]:
        for k in [.00075,.0009,.001,.0011,.00125]:
            ref=reference.mean()+amplitude*(reference-reference.mean())
            variants.append((f'ref_amplitude{amplitude}_k{k}',k,ref,'reference amplitude sensitivity only; preserve mean; not empirical calibration'))
    for offset in [-1,-.5,.5,1,1.5]:
        for k in [.0009,.001,.0011,.00125,.0015]:
            variants.append((f'ref_offset{offset}_k{k}',k,reference+offset,'uniform reference offset; not fitted hour-by-hour; not empirical calibration'))
    summaries=[]
    for name,k,ref,note in variants:
        vals=response_grid(env,case,prices,z,k,ref)
        parameter=dict(name=name,curvature_mean=k,curvature_std=k*.0828,reference=ref.tolist(),note=note)
        summaries.append(summarize(env,case,prices,vals,parameter))
    original=response_grid(env,case,prices,z,.001,reference)
    matched_tou_index=np.rint((env.profile['retail_price']-26.5)/.05).astype(int)
    nominal_delta=(original['user'].mean(1,dtype=np.float64)[np.arange(24),matched_tou_index]/case['baseline_load']-1)*100
    # Approximate local response slopes around reference and exact finite-grid profit preference.
    slope=[]
    for h in range(24):
        refidx=int(np.argmin(abs(prices-reference[h])))
        a=max(0,refidx-1);b=min(120,refidx+1)
        d=original['user'][h].mean(0,dtype=np.float64)
        slope.append(float(-(d[b]-d[a])/(prices[b]-prices[a])))
    profile_rows=[]
    for h in range(24):
        profile_rows.append(dict(hour=h,baseline_load=float(case['baseline_load'][h]),grid_price=float(case['grid_price'][h]),
                                 tou_price=float(env.profile['retail_price'][h]),response_reference=float(reference[h]),
                                 response_at_tou_load_change_pct=float(nominal_delta[h]),local_demand_slope_at_reference=slope[h],
                                 local_absolute_price_elasticity_at_reference=float(slope[h]*reference[h]/case['baseline_load'][h])))
    inversions=[]
    for a,b in itertools.combinations(range(24),2):
        if env.profile['retail_price'][a]!=env.profile['retail_price'][b] and (env.profile['retail_price'][a]-env.profile['retail_price'][b])*(case['grid_price'][a]-case['grid_price'][b])<0:
            inversions.append([a,b])
    result=dict(scope='diagnostic Monte Carlo expected hourly profit maximization, storage off; unchanged formula; not learned policy',
                prices=prices.tolist(),response_realizations=256,response_seed_start=940000000,nominal_wind_reference_seed=20260914,
                kernel_max_response_abs_error=max(errors),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'eqnsac/environment.py',ROOT/'data/environment.json',ROOT/'data/profile.json']},
                profile=profile_rows,tou_wholesale_pair_order_conflicts=inversions,
                baseline_tou_mean_load_change_pct=float(nominal_delta.mean()),
                baseline_tou_low_mean_load_change_pct=float(nominal_delta[env.profile['retail_price']==26.5].mean()),
                baseline_tou_high_mean_load_change_pct=float(nominal_delta[env.profile['retail_price']==32.5].mean()),
                variants=summaries,
                caveats=['MC mean is taken before choosing price; this is not a hindsight optimum for individual response noise.',
                         'Nominal load/grid profile and one fixed wind realization. Does not prove a jointly optimal storage policy.',
                         '0–5% is a diagnostic definition of small response, not a user-specified acceptance constraint.',
                         'No temporal payback or energy conservation is added: PBDR remains demand increase/decrease, not guaranteed shifting.',
                         'Uniform offsets/flattening diagnose reference sensitivity; none is silently recommended as calibrated customer data.'])
    (OUT/'response_parameter_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    compact=[{k:v for k,v in row.items() if k not in ['parameters','by_hour']}|{'name':row['parameters']['name']} for row in summaries]
    print(json.dumps(dict(kernel_error=max(errors),tou_conflicts=inversions,baseline_tou_low=result['baseline_tou_low_mean_load_change_pct'],baseline_tou_high=result['baseline_tou_high_mean_load_change_pct'],variants=compact),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
