import unittest

import numpy as np
import torch

from eqnsac.agent import Agent, checked_step
from eqnsac.config import Config
from eqnsac.environment import Environment


class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class TrainingSafetyTests(unittest.TestCase):
    def test_bad_gradient_aborts_before_optimizer_mutation(self):
        value = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([value])
        with self.assertRaisesRegex(FloatingPointError, "critic objective=0 member=2"):
            checked_step(BadGradient.apply(value), optimizer, "critic objective=0 member=2")
        self.assertEqual(value.item(), 1.0)
        self.assertEqual(len(optimizer.state), 0)
        self.assertIsNone(value.grad)

    def test_optimizer_accumulator_overflow_is_detected(self):
        value = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([value])
        with self.assertRaisesRegex(FloatingPointError, "optimizer state"):
            checked_step(value * 1e30, optimizer, "actor objective=1")

    def test_policy_rejects_bad_state_and_bad_network(self):
        env = Environment()
        agent = Agent(Config(hidden_dim=16, embedding_dim=8, device="cpu"), env)
        state = env.reset(env.generator.sample(61))
        with self.assertRaisesRegex(FloatingPointError, "19-element state"):
            agent.act(state[:10])
        invalid = state.copy()
        invalid[4] = np.nan
        with self.assertRaisesRegex(FloatingPointError, "19-element state"):
            agent.act(invalid)
        with torch.no_grad():
            agent.actors[1].mean.bias.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "policy action"):
            agent.act(state)


if __name__ == "__main__":
    unittest.main()
