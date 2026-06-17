"""End-to-end smoke test for independent checkpoint evaluation."""

from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluate import main as evaluate_main
from evaluate import run_evaluation
from train import main as train_main


class EvaluateSmokeTest(unittest.TestCase):
    """Train briefly, load the best checkpoint, and evaluate independently."""

    def test_run_evaluation_generates_separate_output_directory(self) -> None:
        """Generate all evaluation artifacts from a real best checkpoint."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "outputs"
            checkpoints_root = root / "checkpoints"
            config_path = root / "evaluation_smoke.yaml"
            config = self._build_config(
                csv_path,
                outputs_root,
                checkpoints_root,
            )
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            training_paths = train_main(config_path)
            self.assertTrue(training_paths.best_model_path.is_file())

            evaluation_paths = run_evaluation(
                config_path,
                training_paths.best_model_path,
            )

            self.assertTrue(
                evaluation_paths.experiment_dir.name.startswith("evaluation_")
            )
            self.assertIn("evaluation_smoke_test", evaluation_paths.experiment_dir.name)
            self.assertNotEqual(
                training_paths.experiment_dir,
                evaluation_paths.experiment_dir,
            )
            required_files = [
                evaluation_paths.metrics_path,
                evaluation_paths.prediction_results_path,
                evaluation_paths.lct_parameters_path,
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

            metrics = json.loads(
                evaluation_paths.metrics_path.read_text(encoding="utf-8")
            )
            self.assertIn("mae", metrics)
            self.assertIn("test_loss", metrics)

            summary = evaluation_paths.summary_path.read_text(encoding="utf-8")
            self.assertIn(str(training_paths.best_model_path), summary)
            self.assertIn("checkpoint_epoch", summary)

            index_path = outputs_root / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            evaluation_rows = [
                row for row in index_rows
                if row["run_dir"] == str(evaluation_paths.experiment_dir)
            ]
            self.assertEqual(len(evaluation_rows), 1)
            self.assertEqual(evaluation_rows[0]["prefix"], "evaluation")
            self.assertEqual(
                evaluation_rows[0]["experiment_name"],
                "evaluation_smoke_test",
            )

    def test_main_requires_checkpoint_path(self) -> None:
        """Reject evaluation without an explicit trained checkpoint."""
        with self.assertRaisesRegex(
            ValueError,
            "checkpoint_path is required for evaluation",
        ):
            evaluate_main(checkpoint_path=None)

    def test_close_evaluation_exports_original_price_scale(self) -> None:
        """Evaluate a close checkpoint and export predictions on raw price scale."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "outputs"
            checkpoints_root = root / "checkpoints"
            config_path = root / "close_evaluation_smoke.yaml"
            config = self._build_config(
                csv_path,
                outputs_root,
                checkpoints_root,
            )
            config["data"]["target_type"] = "close"
            config["experiment"]["name"] = "close_evaluation_smoke_test"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            training_paths = train_main(config_path)
            evaluation_paths = run_evaluation(
                config_path,
                training_paths.best_model_path,
            )

            with evaluation_paths.prediction_results_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                rows = list(csv.DictReader(file))
            y_true_values = [float(row["y_true"]) for row in rows]
            self.assertGreater(min(y_true_values), 100.0)
            self.assertGreater(max(y_true_values), 108.0)

            metrics = json.loads(
                evaluation_paths.metrics_path.read_text(encoding="utf-8")
            )
            self.assertIsNone(metrics["directional_accuracy"])

    @staticmethod
    def _build_config(
        csv_path: Path,
        outputs_root: Path,
        checkpoints_root: Path,
    ) -> dict:
        """Return a small CPU configuration for train-and-evaluate smoke testing."""
        return {
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
                "name": "evaluation_smoke_test",
                "save_outputs": True,
                "outputs_root": str(outputs_root),
                "checkpoints_root": str(checkpoints_root),
            },
        }

    @staticmethod
    def _write_ohlcv_csv(csv_path: Path) -> None:
        """Write a deterministic 120-row OHLCV time series."""
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
