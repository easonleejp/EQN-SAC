"""Hand-calculated checks separating original EQN statistics from Tail extensions."""

import unittest

import torch

from eqnsac.agent import UPDATE_METRIC_NAMES
from eqnsac.config import Config
from eqnsac.risk import midpoint_quantiles, risk_statistics, risk_value
from eqnsac.runner import COLUMNS


class RiskDefinitionTests(unittest.TestCase):
    def test_equal_means_can_have_nonzero_tail_disagreement(self):
        # The members agree on E[Z]=0, but differ in the lower-tail quantile.
        quantiles = torch.tensor([[[-1.0, 1.0]], [[-2.0, 2.0]]])
        taus = midpoint_quantiles(1, 2, "cpu")
        cfg = Config(arm="tail_only", tail_fraction=0.5, tail_lambda=0.1)
        result = risk_statistics(quantiles, taus, cfg)
        torch.testing.assert_close(result.aleatoric_std, torch.tensor([1.5]))
        torch.testing.assert_close(result.epistemic_std, torch.tensor([0.0]))
        torch.testing.assert_close(result.tail_disagreement, torch.tensor([1.0]))
        torch.testing.assert_close(result.policy_value, torch.tensor([-0.1]))
        # The voltage paper's E_k Var_tau differs from original EQN Var_tau E_k.
        voltage_variance = quantiles.var(-1, unbiased=False).mean(0)
        torch.testing.assert_close(voltage_variance, torch.tensor([2.5]))
        torch.testing.assert_close(result.aleatoric_std.square(), torch.tensor([2.25]))

    def test_member_location_disagreement_is_epistemic(self):
        quantiles = torch.tensor([[[1.0, 1.0]], [[3.0, 3.0]]])
        result = risk_statistics(quantiles, midpoint_quantiles(1, 2, "cpu"), Config())
        torch.testing.assert_close(result.aleatoric_std, torch.tensor([0.0]))
        torch.testing.assert_close(result.epistemic_std, torch.tensor([1.0]))
        torch.testing.assert_close(result.policy_value, torch.tensor([2.0]))

    def test_adaptive_level_stops_gradient_and_uses_existing_grid_rule(self):
        quantiles = torch.arange(32, dtype=torch.float32).expand(2, 1, -1).clone()
        quantiles.requires_grad_(True)
        taus = midpoint_quantiles(1, 32, "cpu")
        cfg = Config(arm="adaptive_only", adaptive_beta=100, adaptive_min_level=0.1)
        result = risk_statistics(quantiles, taus, cfg)
        self.assertFalse(result.cvar_level.requires_grad)
        torch.testing.assert_close(result.policy_value, torch.tensor([1.0]))
        result.policy_value.sum().backward()
        expected = torch.zeros_like(quantiles)
        expected[:, :, :3] = 1 / 6
        torch.testing.assert_close(quantiles.grad, expected)

    def test_legacy_interface_explicitly_returns_tail_as_third_value(self):
        quantiles = torch.tensor([[[-1.0, 1.0]], [[-2.0, 2.0]]])
        taus = midpoint_quantiles(1, 2, "cpu")
        cfg = Config(tail_fraction=0.5)
        result = risk_statistics(quantiles, taus, cfg)
        value, aleatoric, third, level = risk_value(quantiles, taus, cfg)
        torch.testing.assert_close(value, result.policy_value)
        torch.testing.assert_close(aleatoric, result.aleatoric_std)
        torch.testing.assert_close(third, result.tail_disagreement)
        torch.testing.assert_close(level, result.cvar_level)

    def test_log_schema_exposes_eqn_epistemic_and_tail_separately(self):
        for role in ("price", "storage"):
            self.assertIn(role + "_epistemic_std", UPDATE_METRIC_NAMES)
            self.assertIn(role + "_tail_disagreement", UPDATE_METRIC_NAMES)
            self.assertNotIn(role + "_ue", COLUMNS)
        self.assertEqual(tuple(COLUMNS[7:7 + len(UPDATE_METRIC_NAMES)]), UPDATE_METRIC_NAMES)


if __name__ == "__main__":
    unittest.main()
