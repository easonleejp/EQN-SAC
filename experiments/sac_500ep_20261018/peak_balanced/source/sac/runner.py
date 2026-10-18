"""Bounded pure-SAC experiments with paired validation and hourly dispatch logs."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import scipy
import torch

from eqnsac.config import ROOT
from eqnsac.scenarios import LEVELS
from eqnsac.single_source import sample_case
from .agent import Agent
from .config import load_config, Config
from .replay import Replay
from .task import Task, daily_summary

VERSION = "sac-1"


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    tmp.replace(path)


def source_manifest():
    paths = sorted((ROOT / "sac").glob("*.py")) + [ROOT / "train_sac.py"]
    paths += [ROOT / "eqnsac" / (name + ".py") for name in ("config", "environment", "scenarios", "single_source")]
    paths += sorted((ROOT / "data").glob("*.json"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def evaluate(agent, cfg, count, seed=910000000, hourly=True):
    task = Task(cfg)
    combinations = list(itertools.product(LEVELS, repeat=3))
    cases = []
    for i in range(count):
        case = sample_case(task.env, cfg, seed + i, combinations[i % len(combinations)])
        response_seed = 42 if cfg.response_rng == "paired_fixed" else seed + i + 700000
        obs = task.reset(case, response_seed)
        rows = []
        for _ in range(24):
            obs, _, _, info = task.step(agent.act(obs, deterministic=True))
            rows.append(info)
        summary = daily_summary(rows, task.reference_rows, task.env)
        summary.update(case=i, scenario_seed=seed + i, response_seed=response_seed)
        if hourly:
            summary.update(hourly=rows, reference_hourly=task.reference_rows)
        cases.append(summary)
    metrics = {}
    for name in ("profit", "reference_profit", "storage_net", "reward", "shape_cost", "peak_cost", "satisfaction", "charge", "discharge", "excess_energy", "final_soc", "price_mean", "price_upper_fraction", "storage_boundary_fraction"):
        metrics[name] = float(np.mean([c[name] for c in cases]))
    for curve in ("grid", "reference_grid", "user_load", "baseline_load", "net_without_storage"):
        for name in ("peak", "valley", "peak_valley", "energy", "std", "rms_ramp"):
            metrics[curve + "_" + name] = float(np.mean([c["curves"][curve][name] for c in cases]))
    return dict(n=count, seed=seed, deterministic=True, metrics=metrics, cases=cases)


def train(cfg, output):
    cfg.validate()
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(cfg.threads)
    task = Task(cfg)
    agent = Agent(cfg, task.observation_dim)
    replay = Replay(cfg.replay_size, task.observation_dim, cfg.seed + 30001)
    scenario_rng = np.random.default_rng(cfg.seed + 40001)
    explore_rng = np.random.default_rng(cfg.seed + 50001)
    out.mkdir(parents=True)
    hashes = source_manifest()
    for name in hashes:
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    provenance = dict(version=VERSION, hashes=hashes, python=platform.python_version(),
                      torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                      device=cfg.device, threads=cfg.threads,
                      action_coordinates="raw normalized tanh controls; storage mapped by task only",
                      shape_target=task.shape_target,
                      shape_price_scale=task.price_scale, shape_load_scale=task.load_scale)
    write_json(out / "config.json", asdict(cfg))
    write_json(out / "manifest.json", provenance)
    write_json(out / "evaluation_00000.json", evaluate(agent, cfg, cfg.evaluation_cases))
    steps = updates = 0
    started = time.perf_counter()
    columns = ["episode", "profit", "reference_profit", "storage_net", "learning_return", "shape_cost", "peak_cost", "grid_peak", "grid_peak_valley", "grid_std", "final_soc", "price_upper_fraction", "updates", "critic_loss", "actor_loss", "alpha", "entropy", "q1_mean", "target_mean", "seconds"]
    with (out / "training.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for episode in range(1, cfg.episodes + 1):
            scenario_seed = int(scenario_rng.integers(1, 2**31 - 1))
            levels = scenario_rng.choice(LEVELS, 3, p=cfg.level_probabilities)
            case = sample_case(task.env, cfg, scenario_seed, levels)
            obs = task.reset(case, 42 if cfg.response_rng == "paired_fixed" else scenario_seed + 700000)
            rows = []
            metrics = {}
            try:
                for hour in range(24):
                    raw = explore_rng.uniform(-1, 1, 2).astype(np.float32) if steps < cfg.warmup_steps else agent.act(obs)
                    next_obs, reward, done, info = task.step(raw)
                    replay.add(obs, raw, reward, next_obs, done)
                    obs = next_obs
                    rows.append(info)
                    steps += 1
                    if steps >= cfg.warmup_steps and len(replay) >= cfg.batch_size:
                        metrics = agent.update(replay)
                        updates += 1
            except Exception as exc:
                write_json(out / "failure.json", dict(episode=episode, steps=steps, scenario_seed=scenario_seed, error=repr(exc)))
                raise
            summary = daily_summary(rows, task.reference_rows, task.env)
            row = dict(episode=episode, profit=summary["profit"], reference_profit=summary["reference_profit"],
                       storage_net=summary["storage_net"], learning_return=summary["reward"], shape_cost=summary["shape_cost"], peak_cost=summary["peak_cost"],
                       grid_peak=summary["curves"]["grid"]["peak"], grid_peak_valley=summary["curves"]["grid"]["peak_valley"],
                       grid_std=summary["curves"]["grid"]["std"], final_soc=summary["final_soc"],
                       price_upper_fraction=summary["price_upper_fraction"], updates=updates, seconds=time.perf_counter() - started)
            for name in ("critic_loss", "actor_loss", "alpha", "entropy", "q1_mean", "target_mean"):
                row[name] = metrics.get(name, "")
            writer.writerow(row)
            stream.flush()
            if episode % cfg.log_every == 0 or episode == cfg.episodes:
                print(f"episode={episode} profit={summary['profit']:.2f} peak={row['grid_peak']:.2f} updates={updates} seconds={row['seconds']:.1f}", flush=True)
            if episode % cfg.evaluation_every == 0 or episode == cfg.episodes:
                write_json(out / f"evaluation_{episode:05d}.json", evaluate(agent, cfg, cfg.evaluation_cases))
            if episode % cfg.checkpoint_every == 0 or episode == cfg.episodes:
                payload = dict(version=VERSION, config=asdict(cfg), model=agent.state_dict(),
                               optimizers=agent.optimizer_state(), provenance=provenance,
                               episode=episode, steps=steps, updates=updates)
                temporary = out / "checkpoint.tmp"
                torch.save(payload, temporary)
                temporary.replace(out / f"checkpoint_{episode:05d}.pt")
    write_json(out / "complete.json", dict(episodes=cfg.episodes, steps=steps, updates=updates, wall_seconds=time.perf_counter() - started))
    return agent


def load_agent(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("version") != VERSION:
        raise ValueError("Expected a pure SAC checkpoint")
    cfg = Config(**saved["config"]).validate()
    current = source_manifest()
    # Runner reporting may evolve; model/observation/physics/data must match.
    critical = {k: v for k, v in saved["provenance"]["hashes"].items() if k not in ("sac/runner.py", "train_sac.py")}
    if any(current.get(k) != v for k, v in critical.items()):
        raise ValueError("Checkpoint source/data differs; use its frozen source directory")
    cfg.device = "cpu"
    torch.set_num_threads(cfg.threads)
    agent = Agent(cfg, Task(cfg).observation_dim)
    agent.load_state_dict(saved["model"])
    return agent, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", type=int, default=108)
    p.add_argument("--seed", type=int, default=930000000)
    args = parser.parse_args()
    if args.command == "train":
        train(load_config(args.config), args.output)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.cases <= 0:
            raise ValueError("Cases must be positive")
        agent, cfg = load_agent(args.checkpoint)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, evaluate(agent, cfg, args.cases, args.seed))


if __name__ == "__main__":
    main()
