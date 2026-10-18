"""Hourly quadratic coefficients retain the original response and accounting."""
import copy
from dataclasses import replace
import unittest

import numpy as np

from sac.config import Config
from sac.parameters import make_environment
from sac.scenarios import sample_case
from sac.task import Task
from eqnsac.environment import Environment


def configuration():
    return replace(Config(), observation='dispatch17', scenario_mode='deterministic_day',
                   response_parameters=dict(curvature_mean=.0011, curvature_std=0.,
                                            demand_noise_std=0., participation_noise_std=0.))


def coefficients(cfg):
    env = make_environment(cfg)
    a = np.linspace(-.0009, -.0013, 24)
    reference = np.asarray(env.spec['hourly_demand_reference']) + np.sin(np.arange(24) / 5)
    b = reference - 2 * a * env.profile['baseline_load']
    return dict(a=a.tolist(), b=b.tolist())


class HourlyUtilityTests(unittest.TestCase):
    def test_invalid_or_ambiguous_coefficients_are_rejected(self):
        cfg = configuration()
        good = coefficients(cfg)
        bad = [dict(a=good['a']), dict(a=good['a'], b=good['b'], c=[1.] * 24),
               dict(a=-.001, b=good['b']), dict(a=good['a'][:23], b=good['b'])]
        for value in [True, float('nan'), float('inf'), 0., .001, -.00001]:
            value_set = copy.deepcopy(good)
            value_set['a'][12] = value
            bad.append(value_set)
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(cfg, hourly_utility=values).validate()
        with self.assertRaises(ValueError):
            replace(cfg, hourly_utility=good,
                    response_parameters={**cfg.response_parameters, 'curvature_std':.00001}).validate()

    def test_hourly_tables_match_original_scalar_formula_at_every_hour(self):
        cfg = configuration()
        coeff = coefficients(cfg)
        controlled = make_environment(replace(cfg, hourly_utility=coeff))
        original = make_environment(cfg)
        case = sample_case(original, cfg, 10)
        original.reset(case, 42)
        controlled.reset(case, 42)
        actions = np.random.default_rng(9).uniform(-1, 1, (24, 2))
        for hour, action in enumerate(actions):
            # Independent legacy path with the equivalent scalar coefficients.
            total = float(original.classes[hour].astype(np.float32).sum())
            original.demand.curvature_mean = -coeff['a'][hour]
            original.spec['hourly_demand_reference'][hour] = coeff['b'][hour] + 2 * coeff['a'][hour] * total
            legacy, hourly = original.step(action), controlled.step(action)
            np.testing.assert_array_equal(legacy[0], hourly[0])
            np.testing.assert_array_equal(legacy[1], hourly[1])
            self.assertEqual(legacy[2:], hourly[2:])
            self.assertEqual(original.rng.bit_generator.state, controlled.rng.bit_generator.state)

    def test_omitted_table_is_bitwise_identical_and_preserves_rng(self):
        cfg = configuration()
        original = Environment(spec=make_environment(cfg).spec)
        injected = make_environment(replace(cfg, hourly_utility={}))
        case = sample_case(original, cfg, 1)
        original.reset(case, 123)
        injected.reset(case, 123)
        for action in np.random.default_rng(51).uniform(-1, 1, (24, 2)):
            one, two = original.step(action), injected.step(action)
            np.testing.assert_array_equal(one[0], two[0])
            np.testing.assert_array_equal(one[1], two[1])
            self.assertEqual(one[2:], two[2:])
            self.assertEqual(original.rng.bit_generator.state, injected.rng.bit_generator.state)

    def test_state_uses_actual_effective_reference_without_future_data(self):
        cfg = configuration()
        coeff = coefficients(cfg)
        task = Task(replace(cfg, hourly_utility=coeff))
        case = sample_case(task.env, cfg, 1)
        case['baseline_load'] = case['baseline_load'] * 1.1
        task.reset(case, 42)
        for hour in range(24):
            total = float(task.env.classes[hour].astype(np.float32).sum())
            expected_ref = coeff['b'][hour] + 2 * coeff['a'][hour] * total
            self.assertAlmostEqual(task.env.effective_demand_reference(hour), expected_ref, places=12)
            obs = task.observe(task.env.state())
            self.assertAlmostEqual(float(obs[8]), 2 * (expected_ref - 26.5) / 6 - 1, places=6)
            future = task.env.classes[hour + 1:].copy()
            task.env.classes[hour + 1:] *= 1.5
            np.testing.assert_array_equal(obs, task.observe(task.env.state()))
            task.env.classes[hour + 1:] = future
            _, reward, _, info = task.step([.1, -.2])
            self.assertEqual(reward, info['profit'] * cfg.reward_scale)

    def test_tables_are_copied_and_do_not_modify_tou_or_limits(self):
        cfg = configuration()
        table = coefficients(cfg)
        default = make_environment(cfg)
        configured = make_environment(replace(cfg, hourly_utility=table))
        np.testing.assert_array_equal(default.profile['retail_price'], configured.profile['retail_price'])
        self.assertEqual(default.spec['user'], configured.spec['user'])
        self.assertEqual(default.spec['storage'], configured.spec['storage'])
        table['a'][0] = -9
        self.assertNotEqual(configured.spec['hourly_utility']['a'][0], -9)


if __name__ == '__main__':
    unittest.main()
