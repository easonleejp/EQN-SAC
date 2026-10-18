"""Explicit training/evaluation commands. Never searches settings or starts four jobs."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from .config import Config, ROOT, load_config
from .environment import Environment
from .agent import Agent
from .replay import Replay
from .scenarios import LEVELS
from .single_source import sample_case


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )


def evaluation(agent, env, count, seed=910000000):
    rows = []
    cases = list(itertools.product(LEVELS, repeat=3))
    for i in range(count):
        case = sample_case(env, agent.cfg, seed + i, cases[i % 27])
        response_seed = (
            42 if agent.cfg.response_rng == "paired_fixed" else seed + i + 700000
        )
        state = env.reset(case, response_seed)
        profit = storage = satisfaction = throughput = 0.0
        for _ in range(24):
            action = agent.act(state, True)
            state, _, done, info = env.step(action)
            profit += info["profit"]
            storage += info["storage_marginal_profit_reward"]
            satisfaction += info["mean_satisfaction"]
            throughput += info["charge"] + info["discharge"]
        rows.append(
            dict(
                case=i,
                profit=profit,
                storage_net=storage,
                satisfaction=satisfaction / 24,
                throughput=throughput,
                final_soc=env.soc,
                scenario_seed=seed + i,
                load_delta=case.get("load_delta", None),
            )
        )
    values = np.array([r["profit"] for r in rows])
    n = max(1, int(np.ceil(count * 0.1)))
    return dict(
        n=count,
        seed=seed,
        scenario_mode=agent.cfg.scenario_mode,
        equal_combination_weight=(agent.cfg.scenario_mode == "three_source" and count % 27 == 0),
        mean_profit=float(values.mean()),
        cvar10=float(np.sort(values)[:n].mean()),
        mean_storage_net=float(np.mean([r["storage_net"] for r in rows])),
        cases=rows,
    )


def train(cfg, out, resume=None):
    cfg.validate()
    out = Path(out)
    if resume is None:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        raise ValueError("Resume output directory must exist")
    env = Environment(cfg.storage_reward_mix)
    agent = Agent(cfg, env)
    replay = Replay(
        cfg.replay_size, cfg.ensemble_size, cfg.bootstrap_probability, cfg.seed + 30001
    )
    scenario_rng = np.random.default_rng(cfg.seed + 40001)
    explore_rng = np.random.default_rng(cfg.seed + 50001)
    first = 1
    steps = updates = 0
    if resume:
        saved = torch.load(resume, map_location=agent.device, weights_only=False)
        previous = json.loads(json.dumps(saved["config"]))
        current = json.loads(json.dumps(asdict(cfg)))
        if any(
            previous[k] != current[k]
            for k in previous
            if k not in ("episodes", "device")
        ):
            raise ValueError("Resume configuration differs")
        if Path(resume).resolve().parent != out.resolve():
            raise ValueError("Resume checkpoint must belong to output directory")
        agent.load_state_dict(saved["model"])
        agent.load_optimizer_state(saved["optimizers"])
        replay.load_state_dict(saved["replay"])
        scenario_rng.bit_generator.state = saved["scenario_rng"]
        explore_rng.bit_generator.state = saved["explore_rng"]
        first = saved["episode"] + 1
        steps = saved["steps"]
        updates = saved["updates"]
        if cfg.episodes < first:
            raise ValueError("Requested episode budget already completed")
        # Roll back log rows after the last atomic checkpoint after a crash.
        log = out / "training.csv"
        if log.exists():
            with log.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                names = reader.fieldnames
                rows = [row for row in reader if int(row["episode"]) < first]
            with log.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=names)
                writer.writeheader()
                writer.writerows(rows)
        (out / "complete.json").unlink(missing_ok=True)
    write_json(out / "config.json", asdict(cfg))
    write_json(
        out / "manifest.json",
        {
            "version": "clean-1",
            "device": str(agent.device),
            "torch": torch.__version__,
            "sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in ("eqnsac", "data")
                for p in (ROOT / folder).glob("*")
                if p.is_file()
            },
            "notes": "No deployment override, imitation or oracle training; not bitwise reproduction of historical training.",
        },
    )
    columns = [
        "episode",
        "profit",
        "storage_net",
        "satisfaction",
        "final_soc",
        "seconds",
        "updates",
        "critic_loss",
        "price_actor_loss",
        "storage_actor_loss",
        "price_ua",
        "price_ue",
        "price_cvar",
        "storage_ua",
        "storage_ue",
        "storage_cvar",
        "price_alpha",
        "storage_alpha",
        "scenario_seed",
        "load_delta",
    ]
    if cfg.scenario_mode == "load_scale" and not resume:
        reference = sample_case(env, cfg, 910000000)
        reference["baseline_load"] = env.generator.load.copy()
        reference["class_baseline_load"] = env.generator.load[:, None] * env.generator.shares
        reference["load_delta"] = 0.0
        write_json(out / "reference_case.json", {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in reference.items()
        })
        write_json(out / "evaluation_00000.json",
                   evaluation(agent, Environment(cfg.storage_reward_mix), cfg.evaluation_cases))
    started = time.perf_counter()
    with (out / "training.csv").open(
        "a" if resume else "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if not resume:
            writer.writeheader()
        for episode in range(first, cfg.episodes + 1):
            case_seed = int(scenario_rng.integers(1, 2**31 - 1))
            levels = scenario_rng.choice(LEVELS, 3, p=cfg.level_probabilities)
            case = sample_case(env, cfg, case_seed, levels)
            state = env.reset(
                case, 42 if cfg.response_rng == "paired_fixed" else case_seed + 700000
            )
            profit = storage = satisfaction = 0.0
            metrics = None
            for _ in range(24):
                if steps < cfg.warmup_steps:
                    lo, hi = env.feasible_bounds()
                    action = np.array(
                        [explore_rng.uniform(-1, 1), explore_rng.uniform(lo, hi)],
                        np.float32,
                    )
                else:
                    action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                executed = np.array(
                    [action[0], info["executed_storage_action"]], np.float32
                )
                replay.add(state, executed, reward, next_state, done)
                state = next_state
                steps += 1
                profit += info["profit"]
                storage += info["storage_marginal_profit_reward"]
                satisfaction += info["mean_satisfaction"]
                if steps >= cfg.warmup_steps and replay.ready(cfg.batch_size):
                    metrics = agent.update(replay)
                    updates += 1
            values = (
                [float("nan")] * 11
                if metrics is None
                else metrics.detach().cpu().tolist()
            )
            if metrics is not None and not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite training metrics")
            writer.writerow(
                dict(
                    zip(
                        columns,
                        [
                            episode,
                            profit,
                            storage,
                            satisfaction / 24,
                            env.soc,
                            time.perf_counter() - started,
                            updates,
                            *values,
                            case_seed,
                            case.get("load_delta", float("nan")),
                        ],
                    )
                )
            )
            stream.flush()
            if episode % cfg.log_every == 0 or episode == cfg.episodes:
                print(
                    f"episode={episode} profit={profit:.2f} storage={storage:.2f} updates={updates}",
                    flush=True,
                )
            if episode % cfg.evaluation_every == 0:
                write_json(
                    out / f"evaluation_{episode:05d}.json",
                    evaluation(
                        agent, Environment(cfg.storage_reward_mix), cfg.evaluation_cases
                    ),
                )
            if episode % cfg.checkpoint_every == 0 or episode == cfg.episodes:
                payload = {
                    "version": "clean-1",
                    "config": asdict(cfg),
                    "model": agent.state_dict(),
                    "optimizers": agent.optimizer_state(),
                    "replay": replay.state_dict(),
                    "scenario_rng": scenario_rng.bit_generator.state,
                    "explore_rng": explore_rng.bit_generator.state,
                    "episode": episode,
                    "steps": steps,
                    "updates": updates,
                }
                temporary = out / "checkpoint.tmp"
                torch.save(payload, temporary)
                temporary.replace(out / "checkpoint.pt")
    write_json(
        out / "complete.json",
        {
            "episodes": cfg.episodes,
            "steps": steps,
            "updates": updates,
            "wall_seconds": time.perf_counter() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=("original", "adaptive_only", "tail_only", "full"))
    p.add_argument("--episodes", type=int)
    p.add_argument("--resume", type=Path)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", type=int, default=270)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "train":
        cfg = load_config(args.config)
        if args.arm:
            cfg.arm = args.arm
        if args.episodes is not None:
            cfg.episodes = args.episodes
        train(cfg, args.output, args.resume)
    else:
        if args.cases <= 0:
            parser.error("--cases must be positive")
        if args.output.exists():
            raise FileExistsError(args.output)
        saved = torch.load(
            args.checkpoint, map_location=args.device, weights_only=False
        )
        cfg = Config(**saved["config"])
        cfg.device = args.device
        env = Environment(cfg.storage_reward_mix)
        agent = Agent(cfg, env)
        agent.load_state_dict(saved["model"])
        write_json(args.output, evaluation(agent, env, args.cases))


if __name__ == "__main__":
    main()
