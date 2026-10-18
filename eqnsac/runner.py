"""Explicit training/evaluation commands. Never searches settings or starts four jobs."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
import platform
import shutil
from pathlib import Path
import time
import numpy as np
import scipy
import torch
from .config import Config, ROOT, load_config
from .environment import Environment
from .agent import Agent, UPDATE_METRIC_NAMES
from .replay import Replay
from .scenarios import LEVELS
from .single_source import sample_case


CHECKPOINT_VERSION = "clean-2"
# Agent owns the metric order, so logging cannot silently relabel a tensor.
COLUMNS = [
    "episode", "profit", "storage_net", "satisfaction", "final_soc", "seconds", "updates",
    *UPDATE_METRIC_NAMES,
    "scenario_seed", "load_delta", "wind_delta", "price_delta",
]


def build_manifest(device):
    files = [ROOT / "train.py"]
    files.extend(
        p
        for folder in ("eqnsac", "data")
        for p in sorted((ROOT / folder).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    return {
        "version": CHECKPOINT_VERSION,
        "device": str(device),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "notes": "No deployment override, imitation or oracle training; not bitwise reproduction of historical training.",
    }


def verify_provenance(saved, current, *, resume):
    version = saved.get("version")
    if version == "clean-1":
        if resume:
            raise ValueError("Legacy clean-1 checkpoints have no embedded provenance; evaluate them or start a new run, but do not resume")
        return "legacy_unverified"
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {version!r}")
    recorded = saved.get("provenance")
    if not isinstance(recorded, dict) or recorded.get("sha256") != current["sha256"]:
        raise ValueError("Checkpoint source/data hashes differ from current files")
    if recorded.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Invalid checkpoint provenance version")
    if resume:
        if saved.get("device") != current["device"] or recorded.get("device") != current["device"]:
            raise ValueError("Resume requires the same actual device because RNG states are device-specific")
        if any(recorded.get(k) != current[k] for k in ("torch", "numpy", "scipy", "python")):
            raise ValueError("Resume requires the same Python/torch/numpy/scipy versions")
    return "verified_source_and_data"


def prepare_resume_log(out, episode):
    log = out / "training.csv"
    if not log.is_file():
        raise ValueError("Resume requires training.csv covering every checkpoint episode")
    with log.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Resume training.csv columns differ from the current schema")
        all_rows = list(reader)
    rows = []
    for row in all_rows:
        try:
            row_episode = int(row["episode"])
        except (ValueError, TypeError):
            if len(rows) == episode:
                continue  # A partial crash row after the committed checkpoint.
            raise ValueError("Resume training.csv contains an invalid committed episode") from None
        if row_episode <= episode:
            if row_episode != len(rows) + 1 or any(row.get(k) is None for k in COLUMNS):
                raise ValueError("Resume training.csv does not continuously cover the checkpoint")
            rows.append(row)
    if len(rows) != episode:
        raise ValueError("Resume training.csv does not continuously cover the checkpoint")
    future = [
        path for path in out.glob("evaluation_*.json")
        if path.stem.removeprefix("evaluation_").isdigit()
        and int(path.stem.removeprefix("evaluation_")) > episode
    ]
    stale = future + [path for path in (out / "complete.json", out / "failure.json") if path.exists()]
    if len(all_rows) != len(rows) or stale:
        archive = out / "recovery_archive" / f"after_episode_{episode:05d}_{time.time_ns()}"
        archive.mkdir(parents=True)
        shutil.copy2(log, archive / log.name)
        for path in stale:
            shutil.move(str(path), archive / path.name)
        write_json(archive / "recovery.json", {
            "checkpoint_episode": episode,
            "discarded_log_rows": len(all_rows) - len(rows),
            "reason": "Preserved outputs beyond the resumed checkpoint and previous completion/failure evidence",
        })
    if len(all_rows) != len(rows):
        temporary = out / "training.rollback.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(log)


def write_failure(out, *, episode, hour, steps, updates, scenario_seed, state, action, error):
    def safe(value):
        if isinstance(value, np.ndarray):
            return safe(value.tolist())
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return str(value)
        return value
    write_json(out / "failure.json", {
        "episode": episode, "hour": hour, "steps": steps, "updates": updates,
        "scenario_seed": scenario_seed, "state": safe(state), "action": safe(action),
        "error": str(error), "error_type": type(error).__name__,
        "checkpoint_preserved": (out / "checkpoint.pt").exists(),
    })


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8"
    )
    temporary.replace(path)


def evaluation(agent, env, count, seed=910000000):
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Evaluation count must be a positive integer")
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
                wind_delta=case.get("wind_delta", None),
                price_delta=case.get("price_delta", None),
            )
        )
    values = np.array([r["profit"] for r in rows])
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite evaluation profits")
    ordered = np.sort(values)
    tail_mass = count * 0.1
    whole = int(np.floor(tail_mass))
    fraction = tail_mass - whole
    tail_total = ordered[:whole].sum()
    if fraction:
        tail_total += fraction * ordered[whole]
    return dict(
        n=count,
        seed=seed,
        scenario_mode=agent.cfg.scenario_mode,
        equal_combination_weight=(agent.cfg.scenario_mode == "three_source" and count % 27 == 0),
        mean_profit=float(values.mean()),
        cvar10=float(tail_total / tail_mass),
        cvar10_definition="Mean of the lowest 10% of the equally weighted empirical profit distribution; the boundary observation receives fractional weight",
        tail_probability=0.1,
        mean_storage_net=float(np.mean([r["storage_net"] for r in rows])),
        cases=rows,
    )


def train(cfg, out, resume=None):
    cfg.validate()
    out = Path(out)
    if resume is None:
        if out.exists():
            raise FileExistsError(out)
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
    elapsed_before = 0.0
    manifest = build_manifest(agent.device)
    if resume:
        saved = torch.load(resume, map_location=agent.device, weights_only=False)
        verify_provenance(saved, manifest, resume=True)
        existing_manifest = out / "manifest.json"
        if not existing_manifest.is_file() or json.loads(existing_manifest.read_text("utf-8")) != saved["provenance"]:
            raise ValueError("Original manifest is missing or differs from checkpoint provenance")
        previous = json.loads(json.dumps(saved["config"]))
        current = json.loads(json.dumps(asdict(cfg)))
        if previous.keys() != current.keys() or any(
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
        elapsed_before = saved.get("elapsed_seconds")
        if not isinstance(elapsed_before, (int, float)) or not np.isfinite(elapsed_before) or elapsed_before < 0:
            raise ValueError("Checkpoint has no valid cumulative elapsed time")
        prepare_resume_log(out, saved["episode"])
    else:
        out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", asdict(cfg))
    if not resume:
        write_json(out / "manifest.json", manifest)
    columns = COLUMNS
    if cfg.scenario_mode != "three_source" and not resume:
        reference = sample_case(env, cfg, 910000000)
        reference["baseline_load"] = env.generator.load.copy()
        reference["class_baseline_load"] = env.generator.load[:, None] * env.generator.shares
        reference["load_delta"] = 0.0
        if cfg.scenario_mode == "wind_scale":
            reference["wind_power"] = env.generator.sample(cfg.reference_seed, "low", "low", "low")["wind_power"]
        reference["wind_delta"] = 0.0
        reference["grid_price"] = env.generator.price.copy()
        reference["price_delta"] = 0.0
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
            for hour in range(24):
                input_state = np.asarray(state).copy()
                action = None
                try:
                    if steps < cfg.warmup_steps:
                        lo, hi = env.feasible_bounds()
                        action = np.array(
                            [explore_rng.uniform(-1, 1), explore_rng.uniform(lo, hi)],
                            np.float32,
                        )
                    else:
                        action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    if not all(np.isfinite(value).all() for value in (next_state, reward, [info["profit"], info["storage_marginal_profit_reward"], info["mean_satisfaction"]])):
                        raise FloatingPointError("Nonfinite environment transition")
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
                except FloatingPointError as error:
                    write_failure(out, episode=episode, hour=hour, steps=steps,
                                  updates=updates, scenario_seed=case_seed,
                                  state=input_state, action=action, error=error)
                    raise
            values = (
                [float("nan")] * len(UPDATE_METRIC_NAMES)
                if metrics is None
                else metrics.detach().cpu().tolist()
            )
            if metrics is not None and not np.isfinite(values).all():
                error = FloatingPointError("Nonfinite training metrics")
                write_failure(out, episode=episode, hour=hour, steps=steps,
                              updates=updates, scenario_seed=case_seed,
                              state=input_state, action=action, error=error)
                raise error
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
                            elapsed_before + time.perf_counter() - started,
                            updates,
                            *values,
                            case_seed,
                            case.get("load_delta", float("nan")),
                            case.get("wind_delta", float("nan")),
                            case.get("price_delta", float("nan")),
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
                    "version": CHECKPOINT_VERSION,
                    "provenance": manifest,
                    "device": str(agent.device),
                    "elapsed_seconds": elapsed_before + time.perf_counter() - started,
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
            "wall_seconds": elapsed_before + time.perf_counter() - started,
            "wall_seconds_definition": "Cumulative committed training segments; resumed segments start at the saved checkpoint time. Excludes initialization, work lost after that checkpoint, and its final save overhead; not total compute time.",
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
        cfg.validate()
        provenance_status = verify_provenance(saved, build_manifest(args.device), resume=False)
        env = Environment(cfg.storage_reward_mix)
        agent = Agent(cfg, env)
        agent.load_state_dict(saved["model"])
        result = evaluation(agent, env, args.cases)
        result["checkpoint"] = {
            "path": str(args.checkpoint.resolve()),
            "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "version": saved.get("version"),
            "episode": saved.get("episode"),
            "provenance_status": provenance_status,
        }
        result["config"] = asdict(cfg)
        result["evaluation_manifest"] = build_manifest(agent.device)
        write_json(args.output, result)


if __name__ == "__main__":
    main()
