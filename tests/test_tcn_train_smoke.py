"""Smoke test for training a plain TCN in a temporary workspace."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from train import main as train_main


class TCNTrainSmokeTest(unittest.TestCase):
    """Run a tiny plain TCN training flow without touching formal outputs."""

    def test_plain_tcn_training_writes_expected_artifacts(self) -> None:
        """Train a small TCN and verify artifacts, checkpoints, and index row."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "experiments" / "tcn" / "outputs"
            checkpoints_root = root / "experiments" / "tcn" / "checkpoints"
            config_path = root / "tcn_train_smoke.yaml"
            config = self._build_config(csv_path, outputs_root, checkpoints_root)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            paths = train_main(config_path)

            required_files = [
                paths.config_path,
                paths.training_log_path,
                paths.metrics_path,
                paths.prediction_results_path,
                paths.summary_path,
                paths.best_model_path,
                paths.latest_model_path,
                paths.figures_dir / "loss_curve.png",
                paths.figures_dir / "loss_curve.pdf",
                paths.figures_dir / "prediction_curve.png",
                paths.figures_dir / "prediction_curve.pdf",
                paths.figures_dir / "residual_distribution.png",
                paths.figures_dir / "residual_distribution.pdf",
                paths.figures_dir / "error_histogram.png",
                paths.figures_dir / "error_histogram.pdf",
            ]
            for path in required_files:
                self.assertTrue(path.is_file(), "Missing artifact: {}".format(path))
                self.assertGreater(path.stat().st_size, 0)

            self.assertFalse(paths.lct_parameters_path.exists())
            self.assertTrue(paths.experiment_dir.is_relative_to(outputs_root))
            self.assertTrue(paths.checkpoint_dir.is_relative_to(checkpoints_root))
            self.assertFalse(
                str(paths.experiment_dir).replace("\\", "/").find("/lstm/") >= 0
            )
            self.assertEqual(
                paths.experiment_dir.parent,
                outputs_root / "volatility_5" / "baseline_signal_features",
            )

            saved_config = json.loads(paths.config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_config["model"]["type"], "plain_tcn")
            self.assertEqual(saved_config["model"]["channels"], [8, 8])

            summary = paths.summary_path.read_text(encoding="utf-8")
            self.assertIn("model_type: plain_tcn", summary)
            self.assertIn("use_lct_riesz: False", summary)

            metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
            self.assertIn("rmse", metrics)
            self.assertIsNone(metrics["directional_accuracy"])

            with paths.training_log_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                log_rows = list(csv.DictReader(file))
            self.assertEqual(len(log_rows), 1)
            self.assertNotIn("alpha", log_rows[0])

            index_path = root / "experiments" / "tcn" / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            tcn_rows = [
                row for row in index_rows
                if row["run_dir"] == str(paths.experiment_dir)
            ]
            self.assertEqual(len(tcn_rows), 1)
            self.assertEqual(tcn_rows[0]["model_type"], "plain_tcn")
            self.assertEqual(tcn_rows[0]["use_lct_riesz"], "False")

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
                "name": "tcn_volatility_5_baseline_signal_features_smoke",
                "save_outputs": True,
                "outputs_root": str(outputs_root),
                "checkpoints_root": str(checkpoints_root),
            },
        }

    @staticmethod
    def _write_ohlcv_csv(csv_path: Path) -> None:
        """Write deterministic positive OHLCV data for smoke training."""
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
