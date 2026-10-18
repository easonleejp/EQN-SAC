"""Behavioral checks for independent, scalar-reward, joint-action SAC."""

import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sac.agent import Agent
from sac.networks import GaussianActor
from sac.replay import Replay


torch.set_num_threads(1)


def config(**overrides):
    values = dict(hidden_dim=16, device="cpu", seed=42, actor_lr=3e-4,
                  critic_lr=3e-4, alpha_lr=3e-4, initial_alpha=0.2,
                  gamma=1.0, target_tau=0.005, batch_size=8, target_entropy=-2.0)
    values.update(overrides)
    return SimpleNamespace(**values)


def populated_replay(observation_dim=5):
    replay = Replay(64, observation_dim, 2026)
    rng = np.random.default_rng(121)
    for step in range(48):
        replay.add(rng.normal(size=observation_dim), rng.uniform(-1, 1, 2),
                   float(rng.normal()), rng.normal(size=observation_dim), step % 12 == 11)
    return replay


class SACTests(unittest.TestCase):
    def test_joint_actor_log_probability_sums_coordinates(self):
        actor = GaussianActor(5, 16).double()
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
        actions, log_prob = actor.sample(torch.zeros(3, 5, dtype=torch.float64), deterministic=True)
        torch.testing.assert_close(actions, torch.zeros(3, 2, dtype=torch.float64))
        torch.testing.assert_close(log_prob, torch.full((3, 1), -math.log(2 * math.pi), dtype=torch.float64))
        generator = torch.Generator().manual_seed(8)
        actions, log_prob = actor.sample(torch.ones(7, 5, dtype=torch.float64), generator=generator)
        self.assertEqual(tuple(actions.shape), (7, 2))
        self.assertTrue(torch.all(actions.abs() <= 1))
        self.assertTrue(torch.isfinite(log_prob).all())

    def test_target_uses_minimum_joint_entropy_and_terminal_mask(self):
        agent = Agent(config(gamma=0.9), 5)
        with torch.no_grad():
            for network, value in zip(agent.targets, (1.0, 3.0)):
                for parameter in network.parameters():
                    parameter.zero_()
                network.network[-1].bias.fill_(value)
        with patch.object(agent.actor, "sample", return_value=(torch.zeros(2, 2), torch.full((2, 1), -2.0))):
            target = agent._bellman_target(torch.zeros(2, 5), torch.tensor([[5.0], [7.0]]),
                                           torch.tensor([[0.0], [1.0]]), torch.tensor(0.2))
        torch.testing.assert_close(target, torch.tensor([[6.26], [7.0]]))
        self.assertFalse(target.requires_grad)

    def test_actor_update_does_not_modify_critic_parameters(self):
        agent = Agent(config(), 5)
        before_critics = copy.deepcopy(agent.critics.state_dict())
        before_actor = copy.deepcopy(agent.actor.state_dict())
        agent._update_actor(torch.randn(8, 5), torch.tensor(0.2))
        for name, value in agent.critics.state_dict().items():
            self.assertTrue(torch.equal(value, before_critics[name]), name)
        self.assertTrue(any(not torch.equal(value, before_actor[name])
                            for name, value in agent.actor.state_dict().items()))
        self.assertTrue(all(parameter.requires_grad for parameter in agent.critics.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in agent.targets.parameters()))

    def test_deterministic_evaluation_preserves_both_random_streams(self):
        agent = Agent(config(), 5)
        observation = np.ones(5, np.float32)
        before_action = agent.action_rng.get_state().clone()
        before_update = agent.update_rng.get_state().clone()
        first = agent.act(observation, deterministic=True)
        for _ in range(10):
            np.testing.assert_array_equal(first, agent.act(observation, deterministic=True))
        self.assertTrue(torch.equal(before_action, agent.action_rng.get_state()))
        self.assertTrue(torch.equal(before_update, agent.update_rng.get_state()))
        agent.act(observation)
        self.assertFalse(torch.equal(before_action, agent.action_rng.get_state()))
        self.assertTrue(torch.equal(before_update, agent.update_rng.get_state()))

    def test_uniform_replay_wraparound_snapshot_and_sampling_restore(self):
        replay = Replay(16, 5, 23)
        for step in range(25):
            replay.add(np.full(5, step), [0.5, -0.5], float(step), np.full(5, step + 1), step % 4 == 0)
        saved = replay.state_dict()
        replica = Replay(16, 5, 99)
        replica.load_state_dict(saved)
        self.assertEqual(len(replica), 16)
        for _ in range(3):
            a, b = replay.sample(8, "cpu"), replica.sample(8, "cpu")
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), name)
            self.assertTrue(torch.all(a["obs"] >= 9))
            self.assertTrue(torch.all(a["obs"] <= 24))
        before = saved["data"]["obs"].copy()
        replica.add(np.full(5, 100), [0, 0], 0.0, np.zeros(5), False)
        np.testing.assert_array_equal(saved["data"]["obs"], before)
        with self.assertRaises(ValueError):
            replay.add(np.zeros(5), [0, 2], 0.0, np.zeros(5), False)
        with self.assertRaises(ValueError):
            replay.add(np.zeros(5), [0, 0], float("nan"), np.zeros(5), False)

    def test_update_and_full_state_restore_continue_exactly(self):
        cfg = config()
        agent = Agent(cfg, 5)
        replay = populated_replay()
        for _ in range(2):
            metrics = agent.update(replay)
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        replica = Agent(cfg, 5)
        replica.load_state_dict(copy.deepcopy(agent.state_dict()))
        replica.load_optimizer_state(copy.deepcopy(agent.optimizer_state()))
        copied_replay = Replay(64, 5, 999)
        copied_replay.load_state_dict(replay.state_dict())
        observation = np.ones(5, np.float32)
        np.testing.assert_array_equal(agent.act(observation), replica.act(observation))
        original_metrics = agent.update(replay)
        restored_metrics = replica.update(copied_replay)
        self.assertEqual(original_metrics, restored_metrics)
        for name, value in agent.state_dict().items():
            self.assertTrue(torch.equal(value, replica.state_dict()[name]), name)
        self.assertTrue(torch.equal(agent.action_rng.get_state(), replica.action_rng.get_state()))
        self.assertTrue(torch.equal(agent.update_rng.get_state(), replica.update_rng.get_state()))

    def test_replay_restore_rejects_invalid_values_without_mutation(self):
        replay = populated_replay()
        before = replay.state_dict()
        for field, invalid in (("action", 2.0), ("done", 0.5), ("obs", 1e300)):
            with self.subTest(field=field):
                state = copy.deepcopy(before)
                state["data"][field] = state["data"][field].astype(np.float64)
                state["data"][field].flat[0] = invalid
                with self.assertRaises(ValueError):
                    replay.load_state_dict(state)
                after = replay.state_dict()
                for name in ("capacity", "observation_dim", "cursor", "size", "rng"):
                    self.assertEqual(before[name], after[name])
                for name in before["data"]:
                    np.testing.assert_array_equal(before["data"][name], after["data"][name])

    def test_nonfinite_target_stops_before_parameter_updates(self):
        agent = Agent(config(), 5)
        before = copy.deepcopy(agent.state_dict())
        with patch.object(agent.targets[0], "forward", return_value=torch.full((8, 1), float("inf"))):
            with self.assertRaisesRegex(FloatingPointError, "Bellman target"):
                agent.update(populated_replay())
        for name, value in agent.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_nonfinite_policy_input_is_rejected(self):
        agent = Agent(config(), 5)
        with self.assertRaises(FloatingPointError):
            agent.act([0, 0, 0, 0, float("nan")])
        with self.assertRaises(FloatingPointError):
            agent.act(np.zeros(6))


if __name__ == "__main__":
    unittest.main()
