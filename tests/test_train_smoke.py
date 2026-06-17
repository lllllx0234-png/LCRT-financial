"""End-to-end smoke test for the training entry point."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from train import main


class TrainSmokeTest(unittest.TestCase):
    """Run a two-epoch training experiment using temporary OHLCV data."""

    def test_main_generates_complete_experiment_outputs(self) -> None:
        """Train, evaluate, checkpoint, and plot without polluting the project."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "outputs"
            checkpoints_root = root / "checkpoints"
            config_path = root / "smoke_config.yaml"
            config = {
                "data": {
                    "csv_path": str(csv_path),
                    "feature_columns": [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                    ],
                    "target_type": "return",
                    "sequence_length": 10,
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                    "batch_size": 8,
                    "num_workers": 0,
                },
                "model": {
                    "input_dim": 5,
                    "hidden_dim": 8,
                    "lstm_hidden_dim": 16,
                    "num_layers": 1,
                    "output_dim": 1,
                    "dropout": 0.0,
                    "bidirectional": False,
                    "use_lct_riesz": True,
                    "lct_gate_init": 1.0,
                },
                "training": {
                    "epochs": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "seed": 42,
                    "device": "cpu",
                },
                "experiment": {
                    "name": "smoke_test",
                    "save_outputs": True,
                    "outputs_root": str(outputs_root),
                    "checkpoints_root": str(checkpoints_root),
                },
            }
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            paths = main(config_path)

            required_output_files = [
                paths.config_path,
                paths.training_log_path,
                paths.metrics_path,
                paths.prediction_results_path,
                paths.summary_path,
                paths.lct_parameters_path,
                paths.figures_dir / "loss_curve.png",
                paths.figures_dir / "loss_curve.pdf",
                paths.figures_dir / "prediction_curve.png",
                paths.figures_dir / "prediction_curve.pdf",
                paths.figures_dir / "residual_distribution.png",
                paths.figures_dir / "error_histogram.png",
                paths.figures_dir / "lct_parameter_analysis.png",
            ]
            required_checkpoint_files = [
                paths.best_model_path,
                paths.latest_model_path,
            ]
            for path in required_output_files + required_checkpoint_files:
                self.assertTrue(path.is_file(), "Missing artifact: {}".format(path))
                self.assertGreater(path.stat().st_size, 0)

            self.assertTrue(
                paths.experiment_dir.is_relative_to(outputs_root)
            )
            self.assertTrue(
                paths.checkpoint_dir.is_relative_to(checkpoints_root)
            )
            self.assertIn("smoke_test", paths.experiment_dir.name)

            index_path = outputs_root / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            training_rows = [
                row for row in index_rows
                if row["run_dir"] == str(paths.experiment_dir)
            ]
            self.assertEqual(len(training_rows), 1)
            self.assertEqual(training_rows[0]["experiment_name"], "smoke_test")
            self.assertEqual(training_rows[0]["target_type"], "return")
            self.assertEqual(training_rows[0]["model_type"], "lct_riesz_lstm")
            self.assertEqual(training_rows[0]["use_lct_riesz"], "True")

            with paths.training_log_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                log_rows = list(csv.DictReader(file))
            self.assertEqual(len(log_rows), 2)
            self.assertIn("alpha", log_rows[0])
            self.assertIn("gamma", log_rows[0])

            metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
            self.assertIn("mae", metrics)
            self.assertIn("directional_accuracy", metrics)

    def test_close_target_training_exports_original_price_scale(self) -> None:
        """Train close prediction with scaled targets and export raw-price outputs."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "outputs"
            checkpoints_root = root / "checkpoints"
            config_path = root / "close_smoke_config.yaml"
            config = {
                "data": {
                    "csv_path": str(csv_path),
                    "feature_columns": [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                    ],
                    "target_type": "close",
                    "sequence_length": 10,
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                    "batch_size": 8,
                    "num_workers": 0,
                },
                "model": {
                    "input_dim": 5,
                    "hidden_dim": 8,
                    "lstm_hidden_dim": 16,
                    "num_layers": 1,
                    "output_dim": 1,
                    "dropout": 0.0,
                    "bidirectional": False,
                    "use_lct_riesz": True,
                    "lct_gate_init": 1.0,
                },
                "training": {
                    "epochs": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "seed": 42,
                    "device": "cpu",
                },
                "experiment": {
                    "name": "close_smoke_test",
                    "save_outputs": True,
                    "outputs_root": str(outputs_root),
                    "checkpoints_root": str(checkpoints_root),
                },
            }
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            paths = main(config_path)

            self.assertTrue(paths.metrics_path.is_file())
            self.assertTrue(paths.prediction_results_path.is_file())
            self.assertTrue((paths.figures_dir / "prediction_curve.png").is_file())
            self.assertIn("close_smoke_test", paths.experiment_dir.name)

            with paths.prediction_results_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                rows = list(csv.DictReader(file))
            y_true_values = [float(row["y_true"]) for row in rows]
            self.assertGreater(min(y_true_values), 100.0)
            self.assertGreater(max(y_true_values), 108.0)

            metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
            self.assertIsNone(metrics["directional_accuracy"])
            self.assertGreater(metrics["mae"], 1.0)

    def test_log_return_training_generates_directional_metrics(self) -> None:
        """Train log-return prediction in temporary space and keep target scale."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "outputs"
            checkpoints_root = root / "checkpoints"
            config_path = root / "log_return_smoke_config.yaml"
            config = {
                "data": {
                    "csv_path": str(csv_path),
                    "feature_columns": [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                    ],
                    "target_type": "log_return",
                    "sequence_length": 10,
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                    "batch_size": 8,
                    "num_workers": 0,
                },
                "model": {
                    "input_dim": 5,
                    "hidden_dim": 8,
                    "lstm_hidden_dim": 16,
                    "num_layers": 1,
                    "output_dim": 1,
                    "dropout": 0.0,
                    "bidirectional": False,
                    "use_lct_riesz": True,
                    "lct_gate_init": 1.0,
                },
                "training": {
                    "epochs": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "seed": 42,
                    "device": "cpu",
                },
                "experiment": {
                    "name": "log_return_smoke_test",
                    "save_outputs": True,
                    "outputs_root": str(outputs_root),
                    "checkpoints_root": str(checkpoints_root),
                },
            }
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            paths = main(config_path)

            self.assertTrue(paths.metrics_path.is_file())
            self.assertTrue(paths.prediction_results_path.is_file())
            self.assertTrue((paths.figures_dir / "prediction_curve.png").is_file())
            self.assertIn("log_return_smoke_test", paths.experiment_dir.name)

            metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
            self.assertIn("directional_accuracy", metrics)
            self.assertIsNotNone(metrics["directional_accuracy"])

            index_path = outputs_root / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            log_return_rows = [
                row for row in index_rows
                if row["run_dir"] == str(paths.experiment_dir)
            ]
            self.assertEqual(len(log_return_rows), 1)
            self.assertEqual(log_return_rows[0]["target_type"], "log_return")
            self.assertEqual(
                log_return_rows[0]["experiment_name"],
                "log_return_smoke_test",
            )

    @staticmethod
    def _write_ohlcv_csv(csv_path: Path) -> None:
        """Write a deterministic 120-row OHLCV series for smoke training."""
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
