"""Integration checks for plain TCN configuration and model construction."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.models.lstm_forecaster import LCTRieszLSTMForecaster
from src.models.tcn_forecaster import TCNForecaster
from train import build_model, load_config


class TCNIntegrationTest(unittest.TestCase):
    """Validate plain TCN wiring without running formal experiments."""

    def test_build_model_creates_plain_tcn_from_config(self) -> None:
        """Build the formal TCN architecture through the shared model factory."""
        config = load_config(
            "experiments/tcn/configs/tcn_volatility_5_baseline_signal_features.yaml"
        )
        model = build_model(config["model"])

        self.assertIsInstance(model, TCNForecaster)
        self.assertEqual(model.input_dim, 9)
        self.assertEqual(model.channels, (52, 52, 52, 52))
        self.assertEqual(model.kernel_size, 3)
        self.assertEqual(model.output_dim, 1)
        self.assertIs(model.use_lct_riesz, False)
        self.assertIsNone(model.export_lct_parameters())
        self.assertEqual(model.count_parameters(), 59177)

        x = torch.randn(2, 60, 9)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_existing_plain_lstm_config_still_builds(self) -> None:
        """Keep legacy plain LSTM model construction unchanged."""
        config = load_config(
            "experiments/lstm/configs/lstm_volatility_5_baseline_signal_features.yaml"
        )
        model = build_model(config["model"])

        self.assertIsInstance(model, LCTRieszLSTMForecaster)
        self.assertIs(model.use_lct_riesz, False)
        self.assertIsNone(model.export_lct_parameters())

    def test_unknown_model_type_raises_clear_error(self) -> None:
        """Reject unsupported model.type values at the shared factory."""
        with self.assertRaisesRegex(ValueError, "Unsupported model.type: unknown_tcn"):
            build_model({"type": "unknown_tcn"})

    def test_formal_tcn_config_static_properties(self) -> None:
        """Check formal TCN YAML fairness against the LSTM signal baseline."""
        tcn_config_path = Path(
            "experiments/tcn/configs/tcn_volatility_5_baseline_signal_features.yaml"
        )
        lstm_config_path = Path(
            "experiments/lstm/configs/lstm_volatility_5_baseline_signal_features.yaml"
        )
        tcn_config = load_config(tcn_config_path)
        lstm_config = load_config(lstm_config_path)

        self.assertEqual(tcn_config["model"]["type"], "plain_tcn")
        self.assertEqual(tcn_config["model"]["channels"], [52, 52, 52, 52])
        self.assertEqual(tcn_config["data"]["sequence_length"], 60)
        self.assertEqual(tcn_config["data"]["target_type"], "volatility_5")
        self.assertEqual(len(tcn_config["data"]["feature_columns"]), 9)
        self.assertEqual(
            tcn_config["experiment"]["outputs_root"],
            "experiments/tcn/outputs",
        )
        self.assertEqual(
            tcn_config["experiment"]["checkpoints_root"],
            "experiments/tcn/checkpoints",
        )

        self.assertEqual(tcn_config["data"], lstm_config["data"])
        self.assertEqual(tcn_config["training"], lstm_config["training"])
        self.assertEqual(
            tcn_config["experiment"]["save_outputs"],
            lstm_config["experiment"]["save_outputs"],
        )
        self.assertNotEqual(tcn_config["model"], lstm_config["model"])
        self.assertNotEqual(
            tcn_config["experiment"]["name"],
            lstm_config["experiment"]["name"],
        )
        self.assertNotEqual(
            tcn_config["experiment"]["outputs_root"],
            lstm_config["experiment"]["outputs_root"],
        )
        self.assertNotEqual(
            tcn_config["experiment"]["checkpoints_root"],
            lstm_config["experiment"]["checkpoints_root"],
        )


if __name__ == "__main__":
    unittest.main()
