"""Integration checks for plain Transformer configuration and model construction."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.models.lstm_forecaster import LCTRieszLSTMForecaster
from src.models.tcn_forecaster import TCNForecaster
from src.models.transformer_forecaster import TransformerForecaster
from train import build_model, load_config


def _changed_paths(
    left: object,
    right: object,
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    """Return nested mapping/list paths whose values differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[tuple[str, ...]] = set()
        for key in set(left).union(right):
            paths.update(
                _changed_paths(
                    left.get(key),
                    right.get(key),
                    prefix + (str(key),),
                )
            )
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = set()
        for index in range(max(len(left), len(right))):
            left_value = left[index] if index < len(left) else None
            right_value = right[index] if index < len(right) else None
            paths.update(
                _changed_paths(
                    left_value,
                    right_value,
                    prefix + (str(index),),
                )
            )
        return paths
    return set() if left == right else {prefix}


class TransformerIntegrationTest(unittest.TestCase):
    """Validate plain Transformer wiring without formal training."""

    def test_build_model_creates_plain_transformer_from_config(self) -> None:
        """Build the formal Transformer through the shared model factory."""
        config = load_config(
            "experiments/transformer/configs/transformer_volatility_5_baseline_signal_features.yaml"
        )
        model = build_model(config["model"])

        self.assertIsInstance(model, TransformerForecaster)
        self.assertEqual(model.input_dim, 9)
        self.assertEqual(model.d_model, 56)
        self.assertEqual(model.nhead, 4)
        self.assertEqual(model.num_layers, 2)
        self.assertEqual(model.dim_feedforward, 112)
        self.assertEqual(model.max_sequence_length, 60)
        self.assertEqual(model.output_dim, 1)
        self.assertIs(model.causal_attention, True)
        self.assertIs(model.use_lct_riesz, False)
        self.assertIsNone(model.export_lct_parameters())
        self.assertEqual(model.count_parameters(), 55497)

        x = torch.randn(2, 60, 9)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_unknown_model_type_raises_clear_error(self) -> None:
        """Reject unsupported model.type values at the shared factory."""
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported model.type: unknown_transformer",
        ):
            build_model({"type": "unknown_transformer"})

    def test_existing_model_factory_branches_still_build(self) -> None:
        """Keep existing plain LSTM and plain TCN model construction working."""
        lstm_config = load_config(
            "experiments/lstm/configs/lstm_volatility_5_baseline_signal_features.yaml"
        )
        tcn_config = load_config(
            "experiments/tcn/configs/tcn_volatility_5_baseline_signal_features.yaml"
        )

        self.assertIsInstance(build_model(lstm_config["model"]), LCTRieszLSTMForecaster)
        self.assertIsInstance(build_model(tcn_config["model"]), TCNForecaster)

    def test_formal_transformer_config_static_properties(self) -> None:
        """Check formal Transformer YAML properties and output roots."""
        config_path = Path(
            "experiments/transformer/configs/transformer_volatility_5_baseline_signal_features.yaml"
        )
        config = load_config(config_path)
        model = build_model(config["model"])

        self.assertEqual(config["model"]["type"], "plain_transformer")
        self.assertEqual(config["data"]["sequence_length"], 60)
        self.assertEqual(config["model"]["max_sequence_length"], 60)
        self.assertGreaterEqual(
            config["model"]["max_sequence_length"],
            config["data"]["sequence_length"],
        )
        self.assertEqual(config["data"]["target_type"], "volatility_5")
        self.assertEqual(len(config["data"]["feature_columns"]), 9)
        self.assertEqual(
            config["experiment"]["outputs_root"],
            "experiments/transformer/outputs",
        )
        self.assertEqual(
            config["experiment"]["checkpoints_root"],
            "experiments/transformer/checkpoints",
        )
        self.assertEqual(
            config["experiment"]["name"],
            "transformer_volatility_5_baseline_signal_features",
        )
        self.assertIsInstance(model, TransformerForecaster)
        self.assertEqual(model.count_parameters(), 55497)

    def test_transformer_config_matches_lstm_baseline_except_model_and_paths(self) -> None:
        """Ensure Transformer YAML differs from LSTM only in allowed sections."""
        lstm_config = load_config(
            "experiments/lstm/configs/lstm_volatility_5_baseline_signal_features.yaml"
        )
        transformer_config = load_config(
            "experiments/transformer/configs/transformer_volatility_5_baseline_signal_features.yaml"
        )

        self.assertEqual(transformer_config["data"], lstm_config["data"])
        self.assertEqual(transformer_config["training"], lstm_config["training"])
        self.assertNotEqual(
            transformer_config["experiment"]["name"],
            lstm_config["experiment"]["name"],
        )
        self.assertEqual(
            transformer_config["experiment"]["outputs_root"],
            "experiments/transformer/outputs",
        )
        self.assertEqual(
            transformer_config["experiment"]["checkpoints_root"],
            "experiments/transformer/checkpoints",
        )
        self.assertEqual(
            lstm_config["experiment"]["outputs_root"],
            "experiments/lstm/outputs",
        )
        self.assertEqual(
            lstm_config["experiment"]["checkpoints_root"],
            "experiments/lstm/checkpoints",
        )

        changed_paths = _changed_paths(lstm_config, transformer_config)
        unexpected_paths = {
            path
            for path in changed_paths
            if not (
                path[0] == "model"
                or path
                in {
                    ("experiment", "name"),
                    ("experiment", "outputs_root"),
                    ("experiment", "checkpoints_root"),
                }
            )
        }
        self.assertEqual(unexpected_paths, set())

    def test_transformer_causal_attention_rejects_non_bool_config(self) -> None:
        """Avoid bool('false') style parsing for causal_attention."""
        with self.assertRaisesRegex(TypeError, "model.causal_attention must be a bool"):
            build_model(
                {
                    "type": "plain_transformer",
                    "input_dim": 9,
                    "d_model": 56,
                    "nhead": 4,
                    "num_layers": 2,
                    "dim_feedforward": 112,
                    "dropout": 0.1,
                    "max_sequence_length": 60,
                    "output_dim": 1,
                    "causal_attention": "false",
                }
            )


if __name__ == "__main__":
    unittest.main()
