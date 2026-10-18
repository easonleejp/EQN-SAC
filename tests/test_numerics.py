import math
import unittest

import torch
from torch.nn import functional as F

from eqnsac.networks import Actor, quantile_loss


class NumericalTests(unittest.TestCase):
    def test_small_standard_deviation_preserves_sample_log_density(self):
        actor = Actor((0,), 1)
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
            actor.mean.bias.fill_(1)
            actor.log_std.bias.fill_(-20)
        seed = 9
        _, logp = actor.sample(
            torch.ones(4, 1), generator=torch.Generator().manual_seed(seed)
        )
        noise = torch.randn(4, 1, generator=torch.Generator().manual_seed(seed))
        # Float64 provides an independent reference where the small perturbation
        # survives addition to the mean, unlike float32 at this scale.
        z = 1 + math.exp(-20) * noise.double()
        normal = -0.5 * ((z - 1) / math.exp(-20)).square()
        normal = normal + 20 - 0.5 * math.log(2 * math.pi)
        expected = normal - 2 * (math.log(2) - z - F.softplus(-2 * z))
        torch.testing.assert_close(logp, expected.float(), rtol=1e-6, atol=3e-6)
        logp.mean().backward()
        self.assertTrue(all(
            torch.isfinite(p.grad).all() for p in actor.parameters()
            if p.grad is not None
        ))

    def test_huber_large_finite_residual_has_finite_gradient(self):
        prediction = torch.zeros(1, 1, requires_grad=True)
        loss = quantile_loss(
            prediction, torch.tensor([[3e38]]), torch.tensor([[0.5]])
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[-0.5]]))

    def test_huber_matches_piecewise_loss_and_gradient(self):
        prediction = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]], requires_grad=True)
        reference = prediction.detach().clone().requires_grad_()
        target = torch.tensor([[-0.5, 0.0, 0.5]])
        tau = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
        loss = quantile_loss(prediction, target, tau)
        delta = target[:, None, :] - reference[:, :, None]
        huber = torch.where(delta.abs() <= 1, 0.5 * delta.square(), delta.abs() - 0.5)
        expected = ((tau[:, :, None] - (delta.detach() < 0).float()).abs() * huber).sum(1).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        expected.backward()
        torch.testing.assert_close(prediction.grad, reference.grad)


if __name__ == "__main__":
    unittest.main()
