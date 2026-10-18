import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from sac.config import Config, load_config
from sac.parameters import make_environment
from sac.task import Task
from eqnsac.environment import Environment


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

    def test_reward_is_original_profit_and_observation_does_not_change_physics(self):
        tasks = [Task(replace(Config(), observation=mode)) for mode in ('full19', 'causal11', 'market14', 'market18', 'nominal11')]
        for task in tasks:
            task.reset(self.case, 7)
        actions = np.random.default_rng(88).uniform(-1, 1, (24, 2))
        for action in actions:
            results = [task.step(action) for task in tasks]
            for (_, reward, done, info), task in zip(results, tasks):
                self.assertEqual(info, results[0][3])
                self.assertEqual(reward, info['profit'] * task.cfg.reward_scale)
                self.assertEqual(reward, results[0][1])
                self.assertEqual(done, results[0][2])

    def test_causal_observation_does_not_read_future_realizations(self):
        for observation, dim in [('full19', 19), ('causal11', 11), ('market14', 14), ('market18', 18), ('nominal11', 11)]:
            task = Task(replace(Config(), observation=observation))
            task.reset(self.case, 42)
            for hour in range(24):
                state = task.observe(task.env.state())
                # Perturb only unrevealed realizations, preserving every past/current input.
                originals = {key: getattr(task.env, key).copy() for key in ('load', 'price', 'wind')}
                original_case = copy.deepcopy(task.env.case)
                for key in originals:
                    getattr(task.env, key)[hour + 1:] *= 1.5
                for key in ('baseline_load', 'grid_price', 'wind_power'):
                    task.env.case[key][hour + 1:] *= 1.5
                np.testing.assert_array_equal(state, task.observe(task.env.state()))
                self.assertEqual(state.shape, (dim,))
                for key, value in originals.items():
                    setattr(task.env, key, value)
                task.env.case = original_case
                task.step([.2, -.3])

    def test_config_validation(self):
        for kwargs in ({'batch_size':True},{'reward_scale':0},{'episodes':0},{'observation':'future_truth'},
                       {'seed':2**64},{'level_probabilities':[True,0,0]},
                       {'response_parameters':{'price_max':40}}, {'response_parameters':{'curvature_mean':-1}},
                       {'storage_parameters':{'eta_ch':float('nan')}}, {'renewable_penetration':1.1}):
            with self.assertRaises(ValueError):
                replace(Config(), **kwargs).validate()

    def test_parameters_are_isolated_and_default_physics_is_preserved(self):
        original = Environment()
        task = Task(Config())
        self.assertEqual(task.env.spec, original.spec)
        cfg = replace(Config(), response_parameters={'curvature_mean':0.0011},
                      storage_parameters={'p_ch_max':1000, 'p_dis_max':1000}, renewable_penetration=.3)
        task = Task(cfg)
        self.assertEqual(task.env.demand.curvature_mean, .0011)
        self.assertEqual(task.env.storage.p_ch_max, 1000)
        self.assertAlmostEqual(task.env.generator.wind_capacity / original.generator.wind_capacity, 1.5)
        self.assertEqual(task.env.spec['user'], original.spec['user'])
        self.assertEqual(task.env.spec['hourly_demand_reference'], original.spec['hourly_demand_reference'])
        cfg.response_parameters['curvature_mean'] = 9
        self.assertEqual(task.env.demand.curvature_mean, .0011)

    def test_default_parameter_injection_is_stepwise_identical_to_original_environment(self):
        original, injected = Environment(), make_environment(Config())
        np.testing.assert_array_equal(original.reset(self.case, 123), injected.reset(self.case, 123))
        for action in np.random.default_rng(4).uniform(-1, 1, (24, 2)):
            one, two = original.step(action), injected.step(action)
            np.testing.assert_array_equal(one[0], two[0])
            np.testing.assert_array_equal(one[1], two[1])
            self.assertEqual(one[2:], two[2:])

    def test_storage_validation_checks_partial_overrides_against_defaults(self):
        invalid = [dict(soc_min=-10, soc_init=0), dict(soc_init=-1), dict(soc_max=1),
                   dict(soc_min=10000), dict(soc_min=0, soc_init=0, soc_max=0),
                   dict(p_ch_max=0), dict(p_dis_max=-1), dict(eta_ch=2),
                   dict(eta_dis=0), dict(c_es=-.1)]
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                cfg = replace(Config(), storage_parameters=parameters)
                with self.assertRaises(ValueError):
                    cfg.validate()
                with self.assertRaises(ValueError):
                    make_environment(cfg)
        cfg = replace(Config(), storage_parameters=dict(soc_min=0, soc_init=0, soc_max=100,
                      p_ch_max=10, p_dis_max=5, eta_ch=1, eta_dis=1, c_es=0))
        cfg.validate()
        self.assertEqual(make_environment(cfg).storage.soc_min, 0)

    def test_response_truncation_is_allowed_and_tariff_overrides_are_forbidden(self):
        # max(normal draw, min_curvature) remains well-defined when min > mean.
        valid = replace(Config(), response_parameters=dict(curvature_mean=.001, min_curvature=.002))
        valid.validate()
        self.assertEqual(make_environment(valid).demand.min_curvature, .002)
        forbidden = ('price_min', 'price_max', 'retail_price', 'grid_price',
                     'hourly_demand_reference', 'reference_price', 'eta_self', 'dt')
        for parameter in forbidden:
            for field in ('response_parameters', 'storage_parameters'):
                with self.subTest(field=field, parameter=parameter):
                    with self.assertRaises(ValueError):
                        replace(Config(), **{field: {parameter: 1}}).validate()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            for parameter in ('reward', 'shape_weight', 'peak_weight', 'profile', 'price_min', 'reference_price'):
                path.write_text(json.dumps({parameter: 1}))
                with self.subTest(top_level=parameter):
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_reward_scale_is_only_a_scalar_and_does_not_change_profit(self):
        one = Task(Config())
        two = Task(replace(Config(), reward_scale=.0002))
        one.reset(self.case, 333)
        two.reset(self.case, 333)
        for action in np.random.default_rng(11).uniform(-1, 1, (24, 2)):
            _, reward_one, _, info_one = one.step(action)
            _, reward_two, _, info_two = two.step(action)
            self.assertEqual(info_one['profit'], info_two['profit'])
            self.assertEqual(reward_one, info_one['profit'] * one.cfg.reward_scale)
            self.assertEqual(reward_two, info_two['profit'] * two.cfg.reward_scale)

    def test_evaluation_metadata_contains_actual_parameterized_inputs(self):
        from sac.runner import evaluate
        class FixedPolicy:
            @staticmethod
            def act(observation, deterministic=False):
                return np.array([.1, -.5], dtype=np.float32)

        cfg = replace(Config(), observation='market14', renewable_penetration=.3,
                      response_parameters={'curvature_mean':.0011},
                      storage_parameters={'p_ch_max':1000, 'p_dis_max':500})
        result = evaluate(FixedPolicy(), cfg, 1, seed=100)
        actual = make_environment(cfg)
        self.assertEqual(result['metadata']['environment'], actual.spec)
        self.assertEqual(result['metadata']['profile'], {k: v.tolist() for k, v in actual.profile.items()})
        self.assertEqual(result['metadata']['generator']['target_penetration'], .3)
        self.assertEqual(result['metadata']['config']['response_parameters'], cfg.response_parameters)
        self.assertEqual(result['cases'][0]['levels'], dict(load='low', wind='low', price='low'))
        for row in result['cases'][0]['hourly']:
            self.assertEqual(row['learning_reward'], row['profit'] * cfg.reward_scale)
            self.assertNotIn('shape_cost', row)
            self.assertNotIn('peak_cost', row)

    def test_terminal_observation_and_profit_only_after_24_steps(self):
        task = Task(replace(Config(), observation='market14'))
        task.reset(self.case, 42)
        for _ in range(24):
            obs, reward, done, info = task.step([0, 0])
            self.assertEqual(reward, task.cfg.reward_scale * info['profit'])
        self.assertTrue(done)
        np.testing.assert_array_equal(obs, np.zeros(14))


if __name__ == '__main__':
    unittest.main()
