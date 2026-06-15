"""Minimal tests for the LCT-Riesz LSTM financial forecaster."""

from __future__ import annotations

import unittest

import torch

from src.models.lstm_forecaster import LCTRieszLSTMForecaster


class LCTRieszLSTMForecasterTest(unittest.TestCase):
    """Verify enhanced and baseline forecasting modes."""

    def setUp(self) -> None:
        """Create a reproducible OHLCV-shaped input batch."""
        torch.manual_seed(42)
        self.x = torch.randn(4, 60, 5)
        self.model_kwargs = {
            "input_dim": 5,
            "hidden_dim": 16,
            "lstm_hidden_dim": 32,
            "num_layers": 2,
            "output_dim": 1,
            "dropout": 0.1,
            "bidirectional": False,
        }

    def test_lct_riesz_forward_backward_and_export(self) -> None:
        """Run the enhanced model through forward and backward propagation."""
        model = LCTRieszLSTMForecaster(
            **self.model_kwargs,
            use_lct_riesz=True,
            lct_gate_init=1.0,
        )

        output = model(self.x)
        self.assertEqual(output.shape, (4, 1))
        output.mean().backward()

        exported = model.export_lct_parameters()
        self.assertIsNotNone(exported)
        self.assertEqual(
            set(exported or {}),
            {"alpha", "m", "q", "A", "B", "C", "D", "gamma"},
        )
        self.assertGreater(model.count_parameters(), 0)

    def test_plain_lstm_baseline_forward(self) -> None:
        """Run the same input through the plain LSTM baseline."""
        model = LCTRieszLSTMForecaster(
            **self.model_kwargs,
            use_lct_riesz=False,
        )

        output = model(self.x)
        self.assertEqual(output.shape, (4, 1))
        self.assertIsNone(model.export_lct_parameters())
        self.assertGreater(model.count_parameters(), 0)


if __name__ == "__main__":
    unittest.main()
