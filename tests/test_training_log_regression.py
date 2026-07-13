"""Regression tests for complete sequential training logs."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from train import main as train_main


class TrainingLogRegressionTest(unittest.TestCase):
    """Guard against missing, duplicated, or shuffled epoch log rows."""

    def test_training_log_contains_one_ordered_row_per_epoch(self) -> None:
        """Run three temporary epochs and require epoch rows 1, 2, 3."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            epochs = 3
            config_path = root / "training_log_regression.yaml"
            config = self._build_config(
                csv_path=csv_path,
                outputs_root=root / "experiments" / "tcn" / "outputs",
                checkpoints_root=root / "experiments" / "tcn" / "checkpoints",
                epochs=epochs,
            )
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            paths = train_main(config_path)

            with paths.training_log_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                rows = list(csv.DictReader(file))
            logged_epochs = [int(row["epoch"]) for row in rows]

            self.assertEqual(len(rows), epochs)
            self.assertEqual(logged_epochs, [1, 2, 3])
            self.assertEqual(len(set(logged_epochs)), epochs)
            self.assertEqual(logged_epochs, sorted(logged_epochs))

    @staticmethod
    def _build_config(
        csv_path: Path,
        outputs_root: Path,
        checkpoints_root: Path,
        epochs: int,
    ) -> dict:
        """Return a tiny plain TCN configuration for log regression testing."""
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
                "epochs": epochs,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "seed": 42,
                "device": "cpu",
            },
            "experiment": {
                "name": "tcn_training_log_regression",
                "save_outputs": True,
                "outputs_root": str(outputs_root),
                "checkpoints_root": str(checkpoints_root),
            },
        }

    @staticmethod
    def _write_ohlcv_csv(csv_path: Path) -> None:
        """Write deterministic positive OHLCV data for a small smoke run."""
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
