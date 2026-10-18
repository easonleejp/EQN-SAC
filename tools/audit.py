"""Read-only statistical audit and bounded smoke checks; no long training."""

import argparse
import csv, json, sys, time, tempfile
from pathlib import Path
import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
from eqnsac.config import Config
from eqnsac.environment import Environment
from eqnsac.runner import train, evaluation
from eqnsac.agent import Agent

def main():
    parser = argparse.ArgumentParser(description="Environment audit and three-episode smoke test")
    parser.add_argument("--output", type=Path, default=W / "checks/audit")
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = Environment()
    gen = env.generator
    stats = []
    for level in ("low", "medium", "high"):
        cases = [gen.sample(830000 + i, level, level, level) for i in range(300)]
        for source, key, nominal in [
            ("load", "baseline_load", gen.load),
            ("price", "grid_price", gen.price),
            ("wind", "wind_power", np.full(24, gen.wind_capacity * gen.mean_cf)),
        ]:
            x = np.array([c[key] for c in cases])
            ratio = x / nominal
            stats.append(
                dict(
                    level=level,
                    source=source,
                    n_days=300,
                    ratio_mean=float(ratio.mean()),
                    ratio_p01=float(np.quantile(ratio, 0.01)),
                    ratio_p99=float(np.quantile(ratio, 0.99)),
                    ratio_sd=float(ratio.std()),
                    mean_abs_hourly_ramp=float(np.abs(np.diff(x, axis=1)).mean()),
                    daily_total_cv=float(x.sum(1).std() / x.sum(1).mean()),
                )
            )
    with (OUT / "环境统计_900天.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(stats[0]))
        wr.writeheader()
        wr.writerows(stats)
    policies = []
    for level in ("low", "medium", "high"):
        for case_id in range(30):
            case = gen.sample(840000 + case_id, level, level, level)
            env.reset(case)
            profit = storage = 0.0
            for _ in range(24):
                _, _, _, info = env.step([0, 0])
                profit += info["profit"]
                storage += info["storage_marginal_profit_reward"]
            policies.append(
                dict(level=level, case=case_id, profit=profit, storage_net=storage)
            )
    with (OUT / "固定策略_90天.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(policies[0]))
        wr.writeheader()
        wr.writerows(policies)
    result = {
        "environment_stats": stats,
        "fixed_policy": {
            level: {
                "mean": float(
                    np.mean([x["profit"] for x in policies if x["level"] == level])
                ),
                "sd": float(np.std([x["profit"] for x in policies if x["level"] == level])),
            }
            for level in ("low", "medium", "high")
        },
        "minimum_soc_mean_action_power_kw": env.storage.p_ch_max * 0.5,
        "high_any_training": 1 - 0.8**3,
        "high_any_equal_test": 1 - (2 / 3) ** 3,
        "usable_capacity_hours": (env.storage.soc_max - env.storage.soc_min)
        / env.storage.p_ch_max,
    }
    cfg = Config(
        episodes=3,
        hidden_dim=32,
        embedding_dim=16,
        quantiles=16,
        batch_size=8,
        replay_size=128,
        warmup_steps=24,
        device="cpu",
        evaluation_every=2,
        evaluation_cases=1,
        checkpoint_every=1,
        log_every=1,
    )
    with tempfile.TemporaryDirectory(prefix="eqnsac-audit-") as folder:
        folder = Path(folder)
        full = folder / "full"
        split = folder / "split"
        start = time.perf_counter()
        train(cfg, full)
        cfg.episodes = 2
        train(cfg, split)
        cfg.episodes = 3
        train(cfg, split, split / "checkpoint.pt")
        one = torch.load(full / "checkpoint.pt", weights_only=False)
        two = torch.load(split / "checkpoint.pt", weights_only=False)
        exact = all(torch.equal(v, two["model"][k]) for k, v in one["model"].items())
        if not exact:
            raise AssertionError("Resume mismatch")
        agent = Agent(cfg, env)
        agent.load_state_dict(one["model"])
        evaluated = evaluation(agent, env, 3, 920000000)
        result["smoke"] = {
            "episodes": 3,
            "updates": one["updates"],
            "resume_exact": exact,
            "elapsed": time.perf_counter() - start,
            "dimensions": "hidden32 / quantiles16 / CPU; functional test only",
            "evaluation_finite": all(np.isfinite(x["profit"]) for x in evaluated["cases"]),
        }
    (OUT / "核查验证结果.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), "utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
