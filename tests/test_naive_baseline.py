"""Smoke tests for naive close and return baselines."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_naive_baseline import run_naive_baseline


class NaiveBaselineTest(unittest.TestCase):
    """Verify naive baseline outputs without polluting real output folders."""

    def test_close_naive_uses_last_window_close(self) -> None:
        """Close baseline should predict the previous raw Close in the test split."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)
            config_path = self._write_config(
                root,
                csv_path,
                target_type="close",
            )

            paths = run_naive_baseline(config_path)

            self.assertIn("naive_last_close", paths.experiment_dir.name)
            self.assertTrue(paths.metrics_path.is_file())
            self.assertTrue(paths.prediction_results_path.is_file())
            self.assertTrue((paths.figures_dir / "prediction_curve.png").is_file())
            self.assertTrue((paths.figures_dir / "residual_distribution.png").is_file())
            self.assertTrue((paths.figures_dir / "error_histogram.png").is_file())

            with paths.prediction_results_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                rows = list(csv.DictReader(file))

            expected_true, expected_pred = self._expected_close_values()
            self.assertEqual(len(rows), len(expected_true))
            self.assertAlmostEqual(float(rows[0]["y_true"]), expected_true[0])
            self.assertAlmostEqual(float(rows[0]["y_pred"]), expected_pred[0])
            self.assertAlmostEqual(float(rows[-1]["y_pred"]), expected_pred[-1])

            metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
            self.assertIsNone(metrics["directional_accuracy"])
            saved_config = json.loads(paths.config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_config["experiment"]["name"], "naive_last_close")
            self.assertEqual(saved_config["model"]["type"], "naive")
            self.assertEqual(saved_config["naive_rule"], "last_close")
            summary = paths.summary_path.read_text(encoding="utf-8")
            self.assertIn("naive_last_close", summary)
            self.assertIn("last_close", summary)

            index_path = root / "outputs" / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            close_rows = [
                row for row in index_rows
                if row["run_dir"] == str(paths.experiment_dir)
            ]
            self.assertEqual(len(close_rows), 1)
            self.assertEqual(close_rows[0]["model_type"], "naive")
            self.assertEqual(close_rows[0]["naive_rule"], "last_close")

    def test_return_naive_predicts_zero(self) -> None:
        """Return baseline should output exactly zero for every test sample."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)
            config_path = self._write_config(
                root,
                csv_path,
                target_type="return",
            )

            paths = run_naive_baseline(config_path)

            self.assertIn("naive_zero_return", paths.experiment_dir.name)
            self.assertTrue(paths.metrics_path.is_file())
            self.assertTrue(paths.prediction_results_path.is_file())
            with paths.prediction_results_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(all(float(row["y_pred"]) == 0.0 for row in rows))

            metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
            self.assertIn("directional_accuracy", metrics)
            self.assertIsNotNone(metrics["directional_accuracy"])
            saved_config = json.loads(paths.config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_config["experiment"]["name"], "naive_zero_return")
            self.assertEqual(saved_config["model"]["type"], "naive")
            self.assertEqual(saved_config["naive_rule"], "zero_return")

            index_path = root / "outputs" / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            return_rows = [
                row for row in index_rows
                if row["run_dir"] == str(paths.experiment_dir)
            ]
            self.assertEqual(len(return_rows), 1)
            self.assertEqual(return_rows[0]["model_type"], "naive")
            self.assertEqual(return_rows[0]["naive_rule"], "zero_return")

    def test_log_return_naive_predicts_zero(self) -> None:
        """Log-return baseline should output exactly zero and label itself clearly."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)
            config_path = self._write_config(
                root,
                csv_path,
                target_type="log_return",
            )

            paths = run_naive_baseline(config_path)

            self.assertIn("naive_zero_log_return", paths.experiment_dir.name)
            self.assertTrue(paths.metrics_path.is_file())
            self.assertTrue(paths.prediction_results_path.is_file())
            self.assertTrue((paths.figures_dir / "prediction_curve.png").is_file())
            with paths.prediction_results_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(all(float(row["y_pred"]) == 0.0 for row in rows))

            saved_config = json.loads(paths.config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                saved_config["experiment"]["name"],
                "naive_zero_log_return",
            )
            self.assertEqual(saved_config["model"]["type"], "naive")
            self.assertEqual(saved_config["naive_rule"], "zero_log_return")

            metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
            self.assertIsNotNone(metrics["directional_accuracy"])

            index_path = root / "outputs" / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            log_return_rows = [
                row for row in index_rows
                if row["run_dir"] == str(paths.experiment_dir)
            ]
            self.assertEqual(len(log_return_rows), 1)
            self.assertEqual(log_return_rows[0]["model_type"], "naive")
            self.assertEqual(log_return_rows[0]["naive_rule"], "zero_log_return")
            self.assertEqual(log_return_rows[0]["target_type"], "log_return")

    def test_volatility_naive_uses_historical_volatility(self) -> None:
        """Volatility baseline should predict historical five-day log-return std."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)
            config_path = self._write_config(
                root,
                csv_path,
                target_type="volatility_5",
            )

            paths = run_naive_baseline(config_path)

            self.assertIn("naive_historical_volatility_5", paths.experiment_dir.name)
            self.assertTrue(paths.metrics_path.is_file())
            self.assertTrue(paths.prediction_results_path.is_file())
            self.assertTrue((paths.figures_dir / "prediction_curve.png").is_file())

            with paths.prediction_results_path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                rows = list(csv.DictReader(file))
            expected_true, expected_pred = self._expected_volatility_values()
            self.assertEqual(len(rows), len(expected_true))
            self.assertAlmostEqual(float(rows[0]["y_true"]), expected_true[0])
            self.assertAlmostEqual(float(rows[0]["y_pred"]), expected_pred[0])
            self.assertTrue(all(float(row["y_true"]) >= 0.0 for row in rows))
            self.assertTrue(all(float(row["y_pred"]) >= 0.0 for row in rows))

            saved_config = json.loads(paths.config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                saved_config["experiment"]["name"],
                "naive_historical_volatility_5",
            )
            self.assertEqual(saved_config["model"]["type"], "naive")
            self.assertEqual(saved_config["naive_rule"], "historical_volatility_5")

            metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
            self.assertIsNone(metrics["directional_accuracy"])

            index_path = root / "outputs" / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            volatility_rows = [
                row for row in index_rows
                if row["run_dir"] == str(paths.experiment_dir)
            ]
            self.assertEqual(len(volatility_rows), 1)
            self.assertEqual(volatility_rows[0]["model_type"], "naive")
            self.assertEqual(
                volatility_rows[0]["naive_rule"],
                "historical_volatility_5",
            )
            self.assertEqual(volatility_rows[0]["target_type"], "volatility_5")

    @staticmethod
    def _write_config(root: Path, csv_path: Path, target_type: str) -> Path:
        """Write a small config using temporary output/checkpoint roots."""
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
                "target_type": target_type,
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
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "seed": 42,
                "device": "cpu",
            },
            "experiment": {
                "name": "naive_{}_smoke".format(target_type),
                "save_outputs": True,
                "outputs_root": str(root / "outputs"),
                "checkpoints_root": str(root / "checkpoints"),
            },
        }
        config_path = root / "{}_config.yaml".format(target_type)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        return config_path

    @staticmethod
    def _write_ohlcv_csv(csv_path: Path) -> None:
        """Write a deterministic 120-row OHLCV file."""
        rows = 120
        close = 100.0 + np.arange(rows, dtype=np.float64)
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=rows, freq="D"),
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000.0 + np.arange(rows) * 10.0,
            }
        )
        data.to_csv(csv_path, index=False)

    @staticmethod
    def _expected_close_values() -> tuple[list[float], list[float]]:
        """Return expected close targets and predictions for the test split."""
        close = 100.0 + np.arange(120, dtype=np.float64)
        test_close = close[102:]
        y_true = test_close[10:].tolist()
        y_pred = test_close[9:-1].tolist()
        return y_true, y_pred

    @staticmethod
    def _expected_volatility_values() -> tuple[list[float], list[float]]:
        """Return expected future and historical volatility values."""
        close = 100.0 + np.arange(120, dtype=np.float64)
        test_close = close[102:]
        y_true = []
        y_pred = []
        for target_index in range(10, len(test_close) - 4):
            future_log_returns = np.log(
                test_close[target_index : target_index + 5]
                / test_close[target_index - 1 : target_index + 4]
            )
            historical_log_returns = np.log(
                test_close[target_index - 5 : target_index]
                / test_close[target_index - 6 : target_index - 1]
            )
            y_true.append(float(np.std(future_log_returns, ddof=0)))
            y_pred.append(float(np.std(historical_log_returns, ddof=0)))
        return y_true, y_pred


if __name__ == "__main__":
    unittest.main()
