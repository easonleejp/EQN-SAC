"""Information-boundary and physical-invariance checks for causal13."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from sac.config import Config
from sac.parameters import make_environment
from sac.task import Task


class Causal13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = make_environment(Config()).generator.sample(12345)

    def make_task(self, **overrides):
        task = Task(replace(Config(), observation='causal13', **overrides))
        task.env.reset(self.case, 42)
        return task

    def test_prefix_and_same_raw_actions_preserve_full_day_physics_and_reward(self):
        cfg = replace(Config(), observation='causal13')
        cfg.validate()
        causal, current = Task(cfg), Task(replace(cfg, observation='current6'))
        causal_obs = causal.reset(self.case, 42)
        current_obs = current.reset(self.case, 42)
        self.assertEqual(causal.observation_dim, 13)
        self.assertEqual(causal_obs.dtype, np.float32)
        np.testing.assert_array_equal(causal_obs[:6], current_obs)
        actions = np.random.default_rng(181).uniform(-1, 1, (24, 2))
        for hour, raw in enumerate(actions):
            with self.subTest(hour=hour):
                causal_obs, reward, done, info = causal.step(raw)
                current_obs, old_reward, old_done, old_info = current.step(raw)
                np.testing.assert_array_equal(causal_obs[:6], current_obs)
                self.assertEqual(info, old_info)
                self.assertEqual(reward, old_reward)
                self.assertEqual(reward, info['profit'] * cfg.reward_scale)
                self.assertEqual(done, old_done)
        self.assertTrue(done)
        np.testing.assert_array_equal(causal_obs, np.zeros(13, dtype=np.float32))

    def test_fixed_scales_do_not_read_full_day_statistics(self):
        cfg = replace(Config(), observation='causal13')
        env, reference = make_environment(cfg), make_environment(cfg)

        class ForbiddenStatistics:
            def __getitem__(self, key):
                raise AssertionError('Task read full-day profile/normalizers: ' + key)

            def __iter__(self):
                raise AssertionError('Task iterated full-day profile/normalizers')

        profile, normalizers = env.profile, env.norm
        env.profile = env.norm = ForbiddenStatistics()
        # Scenario construction may read its own calibration data; the Task
        # constructor is isolated here to catch observation-scale leakage.
        with patch('sac.task.make_environment', side_effect=[env, reference]):
            task = Task(cfg)
        env.profile, env.norm = profile, normalizers
        env.reset(self.case, 42)
        env.t = 5
        env.price[:] = np.arange(24, dtype=np.float32) + 16
        env.load[5], env.wind[5] = 3750, 1250
        env.soc = env.storage.soc_min + .4 * (env.storage.soc_max - env.storage.soc_min)
        env.norm = ForbiddenStatistics()
        env.generator.wind_capacity = 999999
        obs = task.observe(None)
        np.testing.assert_allclose(obs[:5], [.5, .75, .25, .4, 19 / 24], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(obs[8:11], [0, -.5, -1])

    def test_future_realized_and_nominal_values_cannot_change_current_observation(self):
        for hour in (0, 1, 2, 12, 22):
            with self.subTest(hour=hour):
                task = self.make_task()
                env = task.env
                env.t = hour
                before = task.observe(None)
                for key in ('price', 'load', 'wind'):
                    values = getattr(env, key)
                    values[hour + 1:] = values[hour + 1:] * 3 + 100
                for key in ('grid_price', 'baseline_load', 'wind_power'):
                    env.case[key][hour + 1:] *= 4
                for profile in (env.profile, env.generator.profile):
                    for values in profile.values():
                        if values.ndim and len(values) == 24:
                            values[hour + 1:] = values[hour + 1:] * 5 + 200
                env.generator.price[hour + 1:] *= 6
                env.generator.load[hour + 1:] *= 6
                np.testing.assert_array_equal(before, task.observe(None))

    def test_each_past_price_affects_only_its_lag_and_history_does_not_wrap(self):
        task = self.make_task()
        env = task.env
        env.price[:] = np.arange(24, dtype=np.float32) * 2 + 20
        # Distinct end-of-day prices make accidental negative indexing visible.
        for hour, expected in ((0, [0, 0, 0]), (1, [0, 0, 0]), (2, [1, 0, 0])):
            with self.subTest(initial_hour=hour):
                env.t = hour
                np.testing.assert_array_equal(task.observe(None)[8:11], expected)
        env.t = 6
        before = task.observe(None)
        for lag in (1, 2, 3):
            with self.subTest(lag=lag):
                env.price[env.t - lag] += 4
                expected = before.copy()
                expected[7 + lag] += 2
                np.testing.assert_array_equal(task.observe(None), expected)
                env.price[env.t - lag] -= 4
        # An older price, and nominal rather than measured past prices, are not
        # included in this explicit three-hour history window.
        env.price[:env.t - 3] += 100
        env.profile['grid_price'][:] += 100
        env.generator.profile['grid_price'][:] += 100
        np.testing.assert_array_equal(task.observe(None), before)
        env.price[env.t] += 4
        expected = before.copy()
        expected[0] += 2
        np.testing.assert_array_equal(task.observe(None), expected)

    def test_response_parameters_temperature_and_legacy_state_are_not_inputs(self):
        base = self.make_task()
        changed = self.make_task(response_parameters={
            'curvature_mean': .0014, 'curvature_std': 0,
            'demand_noise_std': 0, 'participation_noise_std': 0,
        }, hourly_utility={'a': [-.0008] * 24, 'b': [38.0] * 24})
        self.assertNotEqual(base.env.effective_demand_reference(12),
                            changed.env.effective_demand_reference(12))
        changed.env.profile['temperature'][:] += 30
        changed.env.generator.profile['temperature'][:] -= 30
        changed.env.previous_price[:] = changed.env.user.price_max
        changed.env.previous_response = 9000
        for hour in (0, 12, 23):
            with self.subTest(hour=hour):
                base.env.t = changed.env.t = hour
                expected = base.observe(None)
                with patch.object(changed.env, 'state', side_effect=AssertionError('legacy state read')), \
                     patch.object(changed.env, 'effective_demand_reference', side_effect=AssertionError('response reference read')), \
                     patch.object(changed.env.generator, 'observations', side_effect=AssertionError('forecast read')):
                    np.testing.assert_array_equal(changed.observe(None), expected)
                    np.testing.assert_array_equal(changed.observe(np.full(19, np.nan)), expected)

    def test_clock_and_feasible_bounds_use_only_current_hour_and_soc(self):
        task = self.make_task(storage_parameters={
            'soc_min': 100, 'soc_init': 600, 'soc_max': 2100,
            'p_ch_max': 1000, 'p_dis_max': 500,
        })
        for hour, clock in ((0, [0, 1]), (6, [1, 0]), (12, [0, -1]), (18, [-1, 0])):
            for energy in (100, 200, 1100, 2050, 2100):
                with self.subTest(hour=hour, energy=energy):
                    task.env.t, task.env.soc = hour, energy
                    obs = task.observe(None)
                    np.testing.assert_allclose(obs[6:8], clock, rtol=0, atol=1e-7)
                    np.testing.assert_allclose(obs[11:13], task.env.feasible_bounds(), rtol=0, atol=1e-7)
                    if energy == 100:
                        np.testing.assert_array_equal(obs[11:13], [0, 1])
                    elif energy == 2100:
                        np.testing.assert_array_equal(obs[11:13], [-1, 0])

    def test_terminal_returns_thirteen_zeros_without_reading_next_day(self):
        task = self.make_task()
        task.env.t = 24
        task.env.price = task.env.load = task.env.wind = np.empty(0)
        task.env.profile = {}
        with patch.object(task.env, 'feasible_bounds', side_effect=AssertionError('terminal bounds read')):
            obs = task.observe(None)
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_array_equal(obs, np.zeros(13, dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
