"""Deterministic environment and exact training/evaluation budget checks."""
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from sac.config import Config
from sac.parameters import make_environment
from sac.task import Task


ZERO_NOISE = dict(curvature_std=0.0, demand_noise_std=0.0, participation_noise_std=0.0)


def fixed_config(**kwargs):
    return replace(Config(), scenario_mode='deterministic_day', observation='nominal11',
                   response_parameters=ZERO_NOISE.copy(), evaluation_every=0,
                   evaluation_cases=1, **kwargs).validate()


class FixedDayTests(unittest.TestCase):
    def test_input_arrays_are_nominal_and_independent_of_seeds_or_levels(self):
        from sac.scenarios import sample_case
        cfg = fixed_config()
        env = make_environment(cfg)
        cases = [sample_case(env, cfg, seed, levels) for seed, levels in
                 [(1, ('low',) * 3), (999, ('high',) * 3)]]
        for key in ('baseline_load', 'grid_price', 'wind_power'):
            np.testing.assert_array_equal(cases[0][key], cases[1][key])
        np.testing.assert_array_equal(cases[0]['baseline_load'], env.profile['baseline_load'])
        np.testing.assert_array_equal(cases[0]['grid_price'], env.profile['grid_price'])
        wind = np.asarray(cases[0]['wind_power'])
        self.assertTrue(np.all(wind == wind[0]))
        self.assertAlmostEqual(wind.sum() / np.asarray(cases[0]['baseline_load']).sum(), .2, places=6)
        self.assertEqual(cases[0]['definition'], cases[1]['definition'])

    def test_all_response_noise_is_zero_even_with_different_rng_seeds(self):
        from sac.scenarios import sample_case
        one, two = Task(fixed_config()), Task(fixed_config())
        case = sample_case(one.env, one.cfg, 1)
        one.reset(case, 42)
        two.reset(case, 9999)
        for action in np.random.default_rng(123).uniform(-1, 1, (24, 2)):
            a, b = one.step(action), two.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1:], b[1:])
            self.assertEqual(a[1], a[3]['profit'] * one.cfg.reward_scale)

    def test_deterministic_mode_rejects_residual_noise(self):
        for name in ZERO_NOISE:
            with self.subTest(name=name):
                params = ZERO_NOISE.copy()
                params[name] = .025
                with self.assertRaises(ValueError):
                    replace(Config(), scenario_mode='deterministic_day', response_parameters=params).validate()
        for field, value in [('updates_per_step', 0), ('updates_per_step', True), ('evaluation_every', -1)]:
            with self.assertRaises(ValueError):
                replace(Config(), **{field: value}).validate()

    def test_no_automatic_evaluation_and_exact_update_budget(self):
        from sac.runner import train

        class SpyAgent:
            updates = 0
            def __init__(self, cfg, dimension):
                self.cfg = cfg
            def act(self, observation, deterministic=False):
                return np.array([0., -.5], dtype=np.float32)
            def update(self, replay):
                type(self).updates += 1
                return {}
            def state_dict(self):
                return {}
            def optimizer_state(self):
                return {}

        cfg = fixed_config(episodes=2, batch_size=1, warmup_steps=0, updates_per_step=3,
                           checkpoint_every=2, log_every=2)
        with tempfile.TemporaryDirectory() as parent:
            out = Path(parent) / 'run'
            with patch('sac.runner.Agent', SpyAgent), patch('sac.runner.evaluate') as evaluation:
                train(cfg, out)
            evaluation.assert_not_called()
            self.assertEqual(SpyAgent.updates, 2 * 24 * 3)
            self.assertEqual(list(out.glob('evaluation_*.json')), [])
            self.assertTrue((out / 'checkpoint_00002.pt').is_file())


if __name__ == '__main__':
    unittest.main()
