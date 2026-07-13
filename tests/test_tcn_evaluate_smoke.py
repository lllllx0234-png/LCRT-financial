"""Smoke test for evaluating a trained plain TCN checkpoint."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluate import run_evaluation
from train import main as train_main


class TCNEvaluateSmokeTest(unittest.TestCase):
    """Train a tiny TCN checkpoint and evaluate it through evaluate.py."""

    def test_plain_tcn_evaluation_loads_checkpoint_and_writes_outputs(self) -> None:
        """Evaluate a real TCN checkpoint without touching formal outputs."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "experiments" / "tcn" / "outputs"
            checkpoints_root = root / "experiments" / "tcn" / "checkpoints"
            config_path = root / "tcn_evaluate_smoke.yaml"
            config = self._build_config(csv_path, outputs_root, checkpoints_root)
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
                str(evaluation_paths.experiment_dir).replace("\\", "/").find("/lstm/")
                >= 0
            )

            metrics = json.loads(
                evaluation_paths.metrics_path.read_text(encoding="utf-8")
            )
            self.assertIn("rmse", metrics)
            self.assertIn("test_loss", metrics)
            self.assertIsNone(metrics["directional_accuracy"])

            summary = evaluation_paths.summary_path.read_text(encoding="utf-8")
            self.assertIn(str(training_paths.best_model_path), summary)
            self.assertIn("model_type: plain_tcn", summary)
            self.assertIn("use_lct_riesz: False", summary)

            index_path = root / "experiments" / "tcn" / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            evaluation_rows = [
                row for row in index_rows
                if row["run_dir"] == str(evaluation_paths.experiment_dir)
            ]
            self.assertEqual(len(evaluation_rows), 1)
            self.assertEqual(evaluation_rows[0]["prefix"], "evaluation")
            self.assertEqual(evaluation_rows[0]["model_type"], "plain_tcn")
            self.assertEqual(evaluation_rows[0]["use_lct_riesz"], "False")

    @staticmethod
    def _build_config(
        csv_path: Path,
        outputs_root: Path,
        checkpoints_root: Path,
    ) -> dict:
        """Return a small plain TCN volatility configuration."""
        signal_features = [
            "log_return",
            "abs_log_return",
            "high_low_range",
            "close_open_return",
        ]
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
                "sequence_length": 10,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "batch_size": 4,
                "num_workers": 0,
            },
            "model": {
                "type": "plain_tcn",
                "input_dim": 9,
                "channels": [8, 8],
                "kernel_size": 3,
                "dropout": 0.0,
                "output_dim": 1,
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "seed": 42,
                "device": "cpu",
            },
            "experiment": {
                "name": "tcn_volatility_5_baseline_signal_features_eval_smoke",
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
