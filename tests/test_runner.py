"""Short, isolated checks for provenance, crash recovery and evaluation statistics."""

import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from eqnsac.config import Config
from eqnsac.runner import (
    COLUMNS,
    Agent,
    build_manifest,
    evaluation,
    main,
    train,
    verify_provenance,
)


torch.set_num_threads(1)


def small_config(**changes):
    cfg = Config(
        episodes=1, hidden_dim=16, embedding_dim=8, quantiles=4,
        ensemble_size=2, batch_size=8, replay_size=128, warmup_steps=24,
        bootstrap_probability=1.0, device="cpu", evaluation_every=100,
        evaluation_cases=1, checkpoint_every=1, log_every=100,
    )
    return replace(cfg, **changes)


def run_quietly(*args, **kwargs):
    with redirect_stdout(io.StringIO()):
        train(*args, **kwargs)


def read_log(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class ProfitEnvironment:
    """Each case returns its index as total day profit."""

    soc = 0.0

    def reset(self, case, seed):
        self.profit = case["profit"]
        self.hour = 0
        return np.zeros(19, np.float32)

    def step(self, action):
        self.hour += 1
        return np.zeros(19, np.float32), [0, 0], self.hour == 24, {
            "profit": self.profit if self.hour == 1 else 0.0,
            "storage_marginal_profit_reward": 0.0,
            "mean_satisfaction": 1.0, "charge": 0.0, "discharge": 0.0,
        }


class RunnerTests(unittest.TestCase):
    def test_fractional_cvar_and_hundred_case_compatibility(self):
        agent = SimpleNamespace(cfg=Config(), act=lambda state, deterministic: [0, 0])
        with patch("eqnsac.runner.sample_case", side_effect=lambda env, cfg, seed, levels: {"profit": seed - 910000000}):
            result = evaluation(agent, ProfitEnvironment(), 27)
            self.assertAlmostEqual(result["cvar10"], (0 + 1 + 0.7 * 2) / 2.7)
            self.assertEqual(result["tail_probability"], 0.1)
            result100 = evaluation(agent, ProfitEnvironment(), 100)
            self.assertEqual(result100["cvar10"], np.arange(100)[:10].mean())
            self.assertEqual(evaluation(agent, ProfitEnvironment(), 1)["cvar10"], 0.0)
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                evaluation(agent, ProfitEnvironment(), count)

    def test_short_resume_is_exact_and_time_is_cumulative(self):
        with tempfile.TemporaryDirectory() as folder:
            full, split = Path(folder) / "full", Path(folder) / "split"
            cfg = small_config(episodes=3)
            run_quietly(cfg, full)
            run_quietly(replace(cfg, episodes=1), split)
            initial = torch.load(split / "checkpoint.pt", weights_only=False)
            manifest_bytes = (split / "manifest.json").read_bytes()
            run_quietly(cfg, split, split / "checkpoint.pt")
            continuous = torch.load(full / "checkpoint.pt", weights_only=False)
            resumed = torch.load(split / "checkpoint.pt", weights_only=False)
            for key, value in continuous["model"].items():
                self.assertTrue(torch.equal(value, resumed["model"][key]), key)
            for name in ("action_rng", "tau_rng"):
                self.assertTrue(torch.equal(continuous["optimizers"][name], resumed["optimizers"][name]))
            self.assertEqual(continuous["steps"], resumed["steps"])
            self.assertEqual(continuous["updates"], resumed["updates"])
            self.assertEqual(manifest_bytes, (split / "manifest.json").read_bytes())
            self.assertGreater(resumed["elapsed_seconds"], initial["elapsed_seconds"])
            rows = read_log(split / "training.csv")
            self.assertEqual([int(row["episode"]) for row in rows], [1, 2, 3])
            self.assertGreater(float(rows[1]["seconds"]), initial["elapsed_seconds"])
            self.assertEqual(list(rows[0]), COLUMNS)
            self.assertIn("price_risk_level", rows[0])
            self.assertNotIn("price_cvar", rows[0])
            for one, two in zip(read_log(full / "training.csv"), rows):
                self.assertEqual({k: v for k, v in one.items() if k != "seconds"},
                                 {k: v for k, v in two.items() if k != "seconds"})

    def test_source_or_data_mismatch_rejects_without_mutating_run(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            cfg = small_config()
            run_quietly(cfg, out)
            before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
            for name in ("eqnsac/runner.py", "data/environment.json"):
                manifest = build_manifest("cpu")
                manifest["sha256"][name] = "changed"
                with patch("eqnsac.runner.build_manifest", return_value=manifest):
                    with self.assertRaisesRegex(ValueError, "hashes"):
                        run_quietly(replace(cfg, episodes=2), out, out / "checkpoint.pt")
                self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()})

    def test_missing_or_incomplete_committed_log_rejects_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            cfg = small_config()
            run_quietly(cfg, out)
            log = out / "training.csv"
            original = log.read_text()
            log.unlink()
            with self.assertRaisesRegex(ValueError, "requires training.csv"):
                run_quietly(replace(cfg, episodes=2), out, out / "checkpoint.pt")
            log.write_text(original.splitlines()[0] + "\n")
            with self.assertRaisesRegex(ValueError, "continuously cover"):
                run_quietly(replace(cfg, episodes=2), out, out / "checkpoint.pt")
            self.assertTrue((out / "complete.json").exists())

    def test_crash_tail_is_archived_before_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            cfg = small_config()
            run_quietly(cfg, out)
            with (out / "training.csv").open("a", encoding="utf-8") as stream:
                stream.write("2,incomplete crash row\n")
                stream.write("partial\n")
            old_log = (out / "training.csv").read_bytes()
            (out / "evaluation_00003.json").write_text('{"stale":true}\n')
            (out / "failure.json").write_text('{"error":"previous failure"}\n')
            run_quietly(replace(cfg, episodes=2), out, out / "checkpoint.pt")
            archives = list((out / "recovery_archive").iterdir())
            self.assertEqual(len(archives), 1)
            archive = archives[0]
            self.assertEqual((archive / "training.csv").read_bytes(), old_log)
            self.assertTrue((archive / "evaluation_00003.json").exists())
            self.assertTrue((archive / "failure.json").exists())
            self.assertTrue((archive / "complete.json").exists())
            self.assertFalse((out / "evaluation_00003.json").exists())
            self.assertFalse((out / "failure.json").exists())
            self.assertEqual([int(row["episode"]) for row in read_log(out / "training.csv")], [1, 2])

    def test_legacy_evaluation_allowed_but_legacy_resume_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            cfg = small_config()
            run_quietly(cfg, out)
            checkpoint = out / "checkpoint.pt"
            saved = torch.load(checkpoint, weights_only=False)
            saved["version"] = "clean-1"
            saved.pop("provenance")
            torch.save(saved, checkpoint)
            with self.assertRaisesRegex(ValueError, "Legacy"):
                run_quietly(replace(cfg, episodes=2), out, checkpoint)
            output = Path(folder) / "evaluation.json"
            argv = ["train.py", "evaluate", "--checkpoint", str(checkpoint),
                    "--output", str(output), "--cases", "1"]
            with patch("sys.argv", argv):
                main()
            result = json.loads(output.read_text())
            self.assertEqual(result["checkpoint"]["provenance_status"], "legacy_unverified")
            self.assertTrue(np.isfinite(result["mean_profit"]))

    def test_version_device_and_manifest_protection(self):
        current = build_manifest("cpu")
        saved = {"version": "clean-2", "provenance": copy.deepcopy(current), "device": "cpu"}
        self.assertEqual(verify_provenance(saved, current, resume=True), "verified_source_and_data")
        saved["device"] = "cuda"
        with self.assertRaisesRegex(ValueError, "same actual device"):
            verify_provenance(saved, current, resume=True)
        saved["device"] = "cpu"
        for package in ("python", "torch", "numpy", "scipy"):
            with self.subTest(package=package):
                altered = copy.deepcopy(saved)
                altered["provenance"][package] = "different"
                with self.assertRaisesRegex(ValueError, "versions"):
                    verify_provenance(altered, current, resume=True)
        self.assertIn("train.py", current["sha256"])
        saved["version"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            verify_provenance(saved, current, resume=False)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            cfg = small_config()
            run_quietly(cfg, out)
            (out / "manifest.json").write_text('{}\n')
            with self.assertRaisesRegex(ValueError, "Original manifest"):
                run_quietly(replace(cfg, episodes=2), out, out / "checkpoint.pt")
            self.assertEqual((out / "manifest.json").read_text(), '{}\n')

    def test_numerical_failure_preserves_checkpoint_and_records_context(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            cfg = small_config()
            run_quietly(cfg, out)
            original_checkpoint = (out / "checkpoint.pt").read_bytes()
            with patch.object(Agent, "update", side_effect=FloatingPointError("critic member 0: nonfinite gradient")):
                with self.assertRaises(FloatingPointError):
                    run_quietly(replace(cfg, episodes=2), out, out / "checkpoint.pt")
            self.assertEqual((out / "checkpoint.pt").read_bytes(), original_checkpoint)
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["episode"], 2)
            self.assertEqual(failure["hour"], 0)
            self.assertEqual(len(failure["state"]), 19)
            self.assertEqual(len(failure["action"]), 2)
            self.assertIn("critic member 0", failure["error"])
            self.assertTrue(failure["checkpoint_preserved"])
            self.assertFalse((out / "complete.json").exists())
            with patch.object(Agent, "update", return_value=torch.full((11,), float("nan"))):
                with self.assertRaisesRegex(FloatingPointError, "Nonfinite training metrics"):
                    run_quietly(replace(cfg, episodes=2), out, out / "checkpoint.pt")
            self.assertEqual((out / "checkpoint.pt").read_bytes(), original_checkpoint)
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["hour"], 23)
            self.assertEqual(failure["error"], "Nonfinite training metrics")

    def test_agent_initialization_failure_leaves_no_run_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            with patch("eqnsac.runner.Agent", side_effect=RuntimeError("CUDA unavailable")):
                with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                    run_quietly(small_config(), out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
