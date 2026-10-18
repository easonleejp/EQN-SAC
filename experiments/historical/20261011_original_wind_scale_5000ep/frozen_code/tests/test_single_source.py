import unittest
import numpy as np
from eqnsac.config import Config
from eqnsac.environment import Environment
from eqnsac.single_source import sample_case


class SingleSourceTests(unittest.TestCase):
    def test_only_wind_changes_and_shape_is_preserved(self):
        env = Environment()
        cfg = Config(scenario_mode="wind_scale")
        reference = env.generator.sample(cfg.reference_seed, "low", "low", "low")
        values = []
        for seed in range(1000):
            case = sample_case(env, cfg, seed)
            values.append(case["wind_delta"])
            self.assertLessEqual(abs(case["wind_delta"]), .05)
            np.testing.assert_allclose(case["wind_power"], reference["wind_power"] * (1 + case["wind_delta"]), rtol=1e-12)
            np.testing.assert_array_equal(case["baseline_load"], env.generator.load)
            np.testing.assert_array_equal(case["grid_price"], env.generator.price)
            self.assertLessEqual(case["wind_power"].max(), env.generator.wind_capacity)
            self.assertEqual(case["load_delta"], 0.0)
        self.assertGreater(np.std(values), .02)
        np.testing.assert_array_equal(sample_case(env, cfg, 15)["wind_power"], sample_case(env, cfg, 15)["wind_power"])

    def test_only_load_changes_and_shape_is_preserved(self):
        env = Environment()
        cfg = Config(scenario_mode="load_scale")
        reference = sample_case(env, cfg, 10)
        deltas = []
        for seed in range(1000):
            case = sample_case(env, cfg, seed)
            deltas.append(case["load_delta"])
            self.assertLessEqual(abs(case["load_delta"]), .05)
            np.testing.assert_allclose(case["baseline_load"] / env.generator.load,
                                       1 + case["load_delta"], rtol=1e-12)
            for key in ("wind_power", "wind_speed", "grid_price"):
                np.testing.assert_array_equal(case[key], reference[key])
            np.testing.assert_allclose(case["class_baseline_load"].sum(axis=1),
                                       case["baseline_load"], rtol=1e-6)
        self.assertGreater(np.std(deltas), .02)
        repeated = sample_case(env, cfg, 10)
        np.testing.assert_array_equal(repeated["baseline_load"], reference["baseline_load"])

    def test_response_randomness_repeats_for_fixed_actions(self):
        env = Environment()
        cfg = Config(scenario_mode="load_scale")
        case = sample_case(env, cfg, 99)
        trajectories = []
        for _ in range(2):
            env.reset(case, 42)
            rows = []
            for hour in range(24):
                state, reward, done, info = env.step(np.zeros(2))
                rows.append(np.concatenate((state, np.asarray(reward).ravel())))
            trajectories.append(rows)
        np.testing.assert_array_equal(trajectories[0], trajectories[1])

    def test_reject_second_random_source(self):
        with self.assertRaises(ValueError):
            Config(scenario_mode="load_scale", response_rng="per_case").validate()
