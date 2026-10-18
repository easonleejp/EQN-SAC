import copy
from dataclasses import replace
import unittest
import numpy as np

from sac.config import Config
from sac.task import Task


class TaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = Task(Config()).env.generator.sample(12345)

    def test_raw_mapping_and_energy_balance(self):
        task = Task(Config())
        task.reset(self.case, 42)
        _, _, _, info = task.step([0, 0])
        # Raw zero is the midpoint of feasible power, NOT an idle battery.
        self.assertAlmostEqual(info['charge'], task.env.storage.p_ch_max / 2)
        expected = info['initial_soc'] + task.env.storage.eta_ch * info['charge'] - info['discharge'] / task.env.storage.eta_dis
        self.assertAlmostEqual(info['soc'], expected)
        balance = info['grid_purchase'] + info['wind'] + info['discharge'] - info['user_load'] - info['charge'] - info['excess_energy']
        self.assertAlmostEqual(balance, 0, places=7)

    def test_centering_changes_reward_without_changing_physics(self):
        base = Task(Config())
        centered = Task(replace(Config(), reward='incremental_profit'))
        np.testing.assert_array_equal(base.reset(self.case, 7), centered.reset(self.case, 7))
        for _ in range(24):
            s1, r1, d1, i1 = base.step([0.2, -0.3])
            s2, r2, d2, i2 = centered.step([0.2, -0.3])
            np.testing.assert_array_equal(s1, s2)
            self.assertEqual(i1['profit'], i2['profit'])
            self.assertAlmostEqual(r1 - r2, i1['reference_profit'] * base.cfg.reward_scale)
            self.assertEqual(d1, d2)

    def test_causal_observation_does_not_read_future_realizations(self):
        task = Task(replace(Config(), observation='causal11'))
        state = task.reset(self.case, 42)
        changed = copy.deepcopy(self.case)
        for key in ('baseline_load', 'grid_price', 'wind_power'):
            changed[key][1:] *= 1.5
        np.testing.assert_array_equal(state, task.reset(changed, 42))
        self.assertEqual(state.shape, (11,))
        self.assertEqual(state[4], 1)
        self.assertAlmostEqual(state[1], task.env.load[0] / task.load_scale)

    def test_shape_penalty_is_explicit_and_separate_from_profit(self):
        cfg = replace(Config(), shape_weight=0.2)
        task = Task(cfg)
        task.reset(self.case, 42)
        _, reward, _, info = task.step([-0.5, 0.7])
        expected = cfg.shape_weight * task.price_scale / task.load_scale * (info['grid_purchase'] - task.shape_target)**2
        self.assertAlmostEqual(info['shape_cost'], expected)
        self.assertAlmostEqual(reward, (info['profit'] - expected) * cfg.reward_scale)

    def test_config_validation(self):
        for kwargs in ({'batch_size':True},{'reward_scale':0},{'shape_weight':-1},{'episodes':0},{'observation':'future_truth'},
                       {'peak_weight':1,'gamma':.99},{'seed':2**64},{'level_probabilities':[True,0,0]}):
            with self.assertRaises(ValueError):
                replace(Config(), **kwargs).validate()

    def test_peak_cost_telescopes_and_observation_includes_running_peak(self):
        cfg = replace(Config(), peak_weight=1.0)
        task = Task(cfg)
        obs = task.reset(self.case, 42)
        self.assertEqual(len(obs), 20)
        self.assertEqual(obs[-1], 0)
        costs = []
        purchases = []
        for _ in range(24):
            obs, _, done, info = task.step([0, 0])
            costs.append(info['peak_cost'])
            purchases.append(info['grid_purchase'])
            expected = 0 if done else max(purchases) / task.load_scale
            self.assertAlmostEqual(float(obs[-1]), expected, places=6)
        self.assertAlmostEqual(sum(costs), cfg.peak_weight * task.price_scale * max(purchases))


if __name__ == '__main__':
    unittest.main()
