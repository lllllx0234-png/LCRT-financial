"""Smoke test for evaluating a trained plain Transformer checkpoint."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from evaluate import run_evaluation
from src.models.transformer_forecaster import TransformerForecaster
from train import build_model, main as train_main


class TransformerEvaluateSmokeTest(unittest.TestCase):
    """Train a tiny Transformer checkpoint and evaluate it through evaluate.py."""

    def test_plain_transformer_evaluation_loads_checkpoint_and_writes_outputs(self) -> None:
        """Evaluate a real Transformer checkpoint without formal outputs."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "experiments" / "transformer" / "outputs"
            checkpoints_root = root / "experiments" / "transformer" / "checkpoints"
            config_path = root / "transformer_evaluate_smoke.yaml"
            config = self._build_config(csv_path, outputs_root, checkpoints_root)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            training_paths = train_main(config_path)
            self.assertTrue(training_paths.best_model_path.is_file())

            saved_config = json.loads(
                training_paths.config_path.read_text(encoding="utf-8")
            )
            model = build_model(saved_config["model"])
            self.assertIsInstance(model, TransformerForecaster)
            checkpoint = torch.load(
                training_paths.best_model_path,
                map_location="cpu",
                weights_only=False,
            )
            load_result = model.load_state_dict(checkpoint["model_state_dict"])
            self.assertEqual(load_result.missing_keys, [])
            self.assertEqual(load_result.unexpected_keys, [])
            self.assertEqual(
                tuple(model.positional_encoding.position_embedding.shape),
                (1, 10, 16),
            )
            self.assertIs(model.causal_attention, True)

            evaluation_paths = run_evaluation(
                config_path,
                training_paths.best_model_path,
            )

            required_files = [
                evaluation_paths.metrics_path,
                evaluation_paths.prediction_results_path,
                evaluation_paths.summary_path,
                evaluation_paths.figures_dir / "prediction_curve.png",
                evaluation_paths.figures_dir / "prediction_curve.pdf",
                evaluation_paths.figures_dir / "residual_distribution.png",
                evaluation_paths.figures_dir / "residual_distribution.pdf",
                evaluation_paths.figures_dir / "error_histogram.png",
                evaluation_paths.figures_dir / "error_histogram.pdf",
            ]
            for path in required_files:
                self.assertTrue(path.is_file(), "Missing artifact: {}".format(path))
                self.assertGreater(path.stat().st_size, 0)

            self.assertFalse(evaluation_paths.lct_parameters_path.exists())
            self.assertTrue(evaluation_paths.experiment_dir.is_relative_to(outputs_root))
            self.assertEqual(
                evaluation_paths.experiment_dir.parent,
                outputs_root / "volatility_5" / "baseline_signal_features",
            )
            self.assertNotEqual(
                training_paths.experiment_dir,
                evaluation_paths.experiment_dir,
            )
            self.assertFalse(
                "/lstm/" in str(evaluation_paths.experiment_dir).replace("\\", "/")
            )
            self.assertFalse(
                "/tcn/" in str(evaluation_paths.experiment_dir).replace("\\", "/")
            )

            metrics = json.loads(
                evaluation_paths.metrics_path.read_text(encoding="utf-8")
            )
            self.assertIn("rmse", metrics)
            self.assertIn("test_loss", metrics)
            self.assertIsNone(metrics["directional_accuracy"])

            summary = evaluation_paths.summary_path.read_text(encoding="utf-8")
            self.assertIn(str(training_paths.best_model_path), summary)
            self.assertIn("model_type: plain_transformer", summary)
            self.assertIn("use_lct_riesz: False", summary)

            index_path = root / "experiments" / "transformer" / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            evaluation_rows = [
                row for row in index_rows
                if row["run_dir"] == str(evaluation_paths.experiment_dir)
            ]
            self.assertEqual(len(evaluation_rows), 1)
            self.assertEqual(evaluation_rows[0]["prefix"], "evaluation")
            self.assertEqual(evaluation_rows[0]["model_type"], "plain_transformer")
            self.assertEqual(evaluation_rows[0]["use_lct_riesz"], "False")

    @staticmethod
    def _build_config(
        csv_path: Path,
        outputs_root: Path,
        checkpoints_root: Path,
    ) -> dict:
        """Return a small plain Transformer volatility configuration."""
        signal_features = [
            "log_return",
            "abs_log_return",
            "high_low_range",
            "close_open_return",
        ]
        sequence_length = 10
        return {
            "data": {
                "csv_path": str(csv_path),
                "feature_columns": [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ] + signal_features,
                "derived_features": signal_features,
                "target_type": "volatility_5",
                "sequence_length": sequence_length,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "batch_size": 4,
                "num_workers": 0,
            },
            "model": {
                "type": "plain_transformer",
                "input_dim": 9,
                "d_model": 16,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.1,
                "max_sequence_length": sequence_length,
                "output_dim": 1,
                "causal_attention": True,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "seed": 42,
                "device": "cpu",
            },
            "experiment": {
                "name": "transformer_volatility_5_baseline_signal_features_eval_smoke",
                "save_outputs": True,
                "outputs_root": str(outputs_root),
                "checkpoints_root": str(checkpoints_root),
            },
        }

    @staticmethod
    def _write_ohlcv_csv(csv_path: Path) -> None:
        """Write deterministic positive OHLCV data for smoke evaluation."""
        rows = 120
        time_index = np.arange(rows, dtype=np.float64)
        close = 100.0 + 0.08 * time_index + np.sin(time_index / 5.0)
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=rows, freq="D"),
                "Open": close - 0.2,
                "High": close + 0.6,
                "Low": close - 0.7,
                "Close": close,
                "Volume": 1_000.0 + 3.0 * time_index,
            }
        )
        data.to_csv(csv_path, index=False)


if __name__ == "__main__":
    unittest.main()
