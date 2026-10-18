import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eqnsac.config import Config, ROOT, load_config
from eqnsac.environment import Environment


class ConfigValidationTests(unittest.TestCase):
    def test_checked_in_configs_remain_valid(self):
        for path in (ROOT / "configs").glob("*.json"):
            with self.subTest(config=path.name):
                load_config(path)

    def test_reject_invalid_types_and_random_seeds(self):
        invalid = (
            {"episodes": 1.5}, {"batch_size": True}, {"warmup_steps": 1.5},
            {"seed": -1}, {"seed": 2**64 - 20001}, {"reference_seed": -1},
            {"seed": 1.0}, {"device": None}, {"price_lr": "0.001"},
            {"reward_scale": True}, {"gamma": float("nan")},
            {"tail_lambda": float("inf")},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Config(**changes).validate()

    def test_reject_invalid_probability_vectors(self):
        for probabilities in (
            [float("nan"), 0.6, 0.4], [0.2, float("inf"), 0.8],
            [True, 0.0, 0.0], ["0.2", 0.6, 0.2], [-0.2, 0.6, 0.6],
            [0.2, 0.6], [0.2, 0.6, 0.3], None,
        ):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                Config(level_probabilities=probabilities).validate()

    def test_reject_non_object_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for value in ([], None, 42):
                path.write_text(json.dumps(value), "utf-8")
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_config(path)


class ScenarioValidationTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment()
        self.case = self.env.generator.sample(918)

    def test_reset_copies_inputs_for_observations_and_accounting(self):
        for dtype in (np.float32, np.float64):
            case = copy.deepcopy(self.case)
            for key in ("baseline_load", "wind_power", "grid_price"):
                case[key] = case[key].astype(dtype)
            reference = Environment()
            reference.reset(copy.deepcopy(case))
            state = self.env.reset(case)
            for key in ("baseline_load", "wind_power", "grid_price"):
                case[key][:] *= 1.5
            with self.subTest(dtype=dtype):
                np.testing.assert_array_equal(self.env.state(), state)
                actual = self.env.step([0.2, 0.7])
                expected = reference.step([0.2, 0.7])
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                self.assertEqual(actual[3], expected[3])

    def test_reject_invalid_arrays_without_changing_active_episode(self):
        self.env.reset(self.case)
        self.env.step([0.2, 0.7])
        before = self.env.state().copy()
        before_rng = copy.deepcopy(self.env.rng.bit_generator.state)
        for key in ("baseline_load", "wind_power", "grid_price"):
            invalid = [np.zeros(23), np.zeros((24, 1)), np.full(24, np.nan),
                       np.full(24, np.inf), np.full(24, 1e100)]
            if key != "grid_price":
                invalid.append(np.full(24, -1.0))
            for value in invalid:
                case = copy.deepcopy(self.case)
                case[key] = value
                with self.subTest(key=key, value=value[0]):
                    with self.assertRaises(ValueError):
                        self.env.reset(case)
                    np.testing.assert_array_equal(self.env.state(), before)
                    self.assertEqual(self.env.t, 1)
                    self.assertEqual(self.env.rng.bit_generator.state, before_rng)

    def test_missing_arrays_and_bad_seed_do_not_replace_episode(self):
        state = self.env.reset(self.case)
        for invalid in (None, {}, {"baseline_load": np.zeros(24)}):
            with self.subTest(case=invalid), self.assertRaises(ValueError):
                self.env.reset(invalid)
            np.testing.assert_array_equal(self.env.state(), state)
        with self.assertRaises(ValueError):
            self.env.reset(self.case, response_seed=-1)
        np.testing.assert_array_equal(self.env.state(), state)

    def test_negative_grid_prices_are_supported(self):
        case = copy.deepcopy(self.case)
        case["grid_price"] *= -1
        self.assertTrue(np.isfinite(self.env.reset(case)).all())
        _, reward, _, info = self.env.step([0, 0])
        self.assertTrue(np.isfinite(reward).all())
        self.assertLessEqual(info["grid_cost"], 0)


if __name__ == "__main__":
    unittest.main()
