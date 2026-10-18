import copy
import json
import unittest
from pathlib import Path
import numpy as np
import torch
from eqnsac.config import Config
from eqnsac.environment import Environment
from eqnsac.networks import QuantileCritic, storage_transform, risk_value, quantile_loss
from eqnsac.agent import Agent
from eqnsac.replay import Replay

torch.set_num_threads(1)


class CoreTests(unittest.TestCase):
    def test_historical_environment_parity(self):
        env = Environment()
        fixtures = json.loads(
            (Path(__file__).parent / "legacy_fixtures.json").read_text("utf-8")
        )
        for fixture in fixtures:
            state = env.reset(fixture["case"])
            for row in fixture["rows"]:
                np.testing.assert_allclose(state, row["state"], rtol=2e-5, atol=2e-5)
                state, _, done, info = env.step(row["action"])
                np.testing.assert_allclose(
                    state, row["next_state"], rtol=2e-5, atol=2e-5
                )
                for key, value in row["info"].items():
                    np.testing.assert_allclose(
                        info[key], value, rtol=2e-5, atol=0.03, err_msg=key
                    )
                self.assertEqual(done, row["done"])

    def test_observation_no_future_realizations(self):
        env = Environment()
        case = env.generator.sample(918)
        state = env.reset(case)
        other = copy.deepcopy(case)
        for key in ("grid_price", "wind_power", "baseline_load"):
            other[key][1:] *= 1.4
        np.testing.assert_array_equal(state, env.reset(other))

    def test_storage_bounds_and_accounting(self):
        env = Environment()
        env.reset(env.generator.sample(311))
        for soc in (
            env.storage.soc_min,
            (env.storage.soc_min + env.storage.soc_max) / 2,
            env.storage.soc_max,
        ):
            env.soc = soc
            state = torch.tensor(env.state())[None]
            for raw in (-1.0, 0.0, 1.0):
                value, _ = storage_transform(state, torch.tensor([[raw]]), env.storage)
                lo, hi = env.feasible_bounds()
                self.assertTrue(lo - 1e-6 <= value.item() <= hi + 1e-6)
        env.reset(env.generator.sample(312))
        for _ in range(24):
            _, _, _, i = env.step([0, 0.7])
            self.assertAlmostEqual(
                i["profit"], i["revenue"] - i["grid_cost"] - i["storage_cost"], places=6
            )

    def test_prior_input_gradient_and_frozen_weights(self):
        torch.manual_seed(918)
        q = QuantileCritic(16, 8, 300).double()
        state = torch.randn(1, 19, dtype=torch.float64)
        action = torch.tensor([[0.17, -0.31]], dtype=torch.float64, requires_grad=True)
        tau = torch.tensor([[0.05, 0.25, 0.75]], dtype=torch.float64)
        result = q(state, action, tau).mean()
        grad = torch.autograd.grad(result, action)[0]
        finite = []
        for dim in range(2):
            delta = torch.zeros_like(action)
            delta[0, dim] = 1e-6
            finite.append(
                (
                    (
                        q(state, action + delta, tau) - q(state, action - delta, tau)
                    ).mean()
                    / 2e-6
                ).item()
            )
        np.testing.assert_allclose(
            grad.detach().numpy()[0], finite, rtol=1e-4, atol=1e-5
        )
        before = {k: v.clone() for k, v in q.prior.state_dict().items()}
        opt = torch.optim.Adam([p for p in q.parameters() if p.requires_grad])
        opt.zero_grad()
        q(state, action, tau).square().mean().backward()
        opt.step()
        for k, v in q.prior.state_dict().items():
            self.assertTrue(torch.equal(before[k], v))

    def test_replay_overwrite_and_restore(self):
        replay = Replay(20, 5, 0.5, 1)
        for n in range(50):
            replay.add(np.full(19, n), [0, 0], [0, 0], np.zeros(19), False)
        for pool in replay.pools:
            self.assertEqual(len(pool), len(set(pool)))
            self.assertTrue(all(0 <= i < 20 for i in pool))
        restored = Replay(20, 5, 0.5, 9)
        restored.load_state_dict(replay.state_dict())
        np.testing.assert_array_equal(
            replay.sample(8, "cpu")["state"], restored.sample(8, "cpu")["state"]
        )

    def test_all_arms_update_and_restore(self):
        for arm in ("original", "adaptive_only", "tail_only", "full"):
            cfg = Config(
                arm=arm,
                hidden_dim=16,
                embedding_dim=8,
                quantiles=8,
                batch_size=8,
                replay_size=100,
                device="cpu",
            )
            env = Environment()
            agent = Agent(cfg, env)
            replay = Replay(100, 5, 0.5, 12)
            state = env.reset(env.generator.sample(410))
            for n in range(60):
                action = agent.act(state)
                ns, r, done, _ = env.step(action)
                replay.add(state, action, r, ns, done)
                state = ns
                if done:
                    state = env.reset(env.generator.sample(411 + n))
            metrics = agent.update(replay)
            self.assertTrue(torch.isfinite(metrics).all())
            replica = Agent(cfg, env)
            replica.load_state_dict(agent.state_dict())
            replica.load_optimizer_state(agent.optimizer_state())
            np.testing.assert_array_equal(agent.act(state), replica.act(state))

    def test_risk_ablation_identity(self):
        z = torch.ones(5, 3, 32) * 7
        tau = ((torch.arange(32) + 0.5) / 32).expand(3, -1)
        for arm in ("original", "adaptive_only", "tail_only", "full"):
            value, ua, ue, _ = risk_value(z, tau, Config(arm=arm))
            torch.testing.assert_close(value, torch.full((3,), 7.0))
            self.assertEqual(ua.sum(), 0)
            self.assertEqual(ue.sum(), 0)

    def test_quantile_loss_scale(self):
        loss = quantile_loss(
            torch.zeros(1, 2), torch.ones(1, 2), torch.tensor([[0.25, 0.75]])
        )
        self.assertAlmostEqual(loss.item(), 0.5)

    def test_candidate_observation(self):
        env = Environment()
        cfg = Config(storage_observation="causal15", hidden_dim=16, device="cpu")
        agent = Agent(cfg, env)
        state = env.reset(env.generator.sample(918))
        self.assertEqual(len(agent.actors[1].indices), 15)
        self.assertTrue(np.isfinite(agent.act(state)).all())


if __name__ == "__main__":
    unittest.main()
