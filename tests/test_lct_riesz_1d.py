"""Minimal tensor tests for the one-dimensional learnable LCT-Riesz module."""

from __future__ import annotations

import unittest

import torch

from src.models.lct_riesz_1d import LearnableLCTRiesz1D


class LearnableLCTRiesz1DTest(unittest.TestCase):
    """Verify shape preservation, gradients, and parameter export."""

    def test_forward_backward_and_parameter_export(self) -> None:
        """Run a complete differentiable pass on an OHLCV-shaped tensor."""
        torch.manual_seed(42)
        x = torch.randn(4, 5, 60)
        module = LearnableLCTRiesz1D(channels=5, gate_init=1.0)

        output = module(x)
        self.assertEqual(output.shape, (4, 5, 60))

        loss = output.mean()
        loss.backward()

        learnable_parameters = {
            "alpha_t": module.lct.alpha_t,
            "log_m_t": module.lct.log_m_t,
            "q_t": module.lct.q_t,
        }
        for name, parameter in learnable_parameters.items():
            self.assertIsNotNone(parameter.grad, f"{name} has no gradient.")
            self.assertTrue(
                torch.isfinite(parameter.grad).all(),
                f"{name} has a non-finite gradient.",
            )

        exported = module.export_parameters()
        expected_keys = {"alpha", "m", "q", "A", "B", "C", "D", "gamma"}
        self.assertEqual(set(exported), expected_keys)
        self.assertTrue(all(isinstance(value, float) for value in exported.values()))
        self.assertAlmostEqual(
            exported["A"] * exported["D"] - exported["B"] * exported["C"],
            1.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
