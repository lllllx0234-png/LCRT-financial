"""Tests for the dual-branch LCT-Riesz LSTM forecaster."""

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

from src.models.dual_branch_lct_lstm import DualBranchLCTRieszLSTMForecaster
from train import main


class DualBranchLCTRieszLSTMTest(unittest.TestCase):
    """Validate the dual-branch model and its training integration."""

    def test_forward_backward_and_parameter_export(self) -> None:
        """Run a minimal differentiable forward pass through both branches."""
        model = DualBranchLCTRieszLSTMForecaster(
            input_dim=9,
            hidden_dim=8,
            lstm_hidden_dim=12,
            signal_feature_indices=[5, 6, 7, 8],
            num_layers=1,
            output_dim=1,
            dropout=0.0,
            bidirectional=False,
        )
        x = torch.randn(4, 16, 9)

        output = model(x)
        self.assertEqual(tuple(output.shape), (4, 1))

        loss = output.mean()
        loss.backward()
        self.assertIsNotNone(model.input_projection.weight.grad)
        self.assertIsNotNone(model.fusion_gate.grad)

        params = model.export_lct_parameters()
        self.assertIsNotNone(params)
        self.assertEqual(
            set(params or {}),
            {"alpha", "m", "q", "A", "B", "C", "D", "gamma"},
        )

    def test_out_of_range_signal_feature_indices_raise_value_error(self) -> None:
        """Reject signal feature indices that cannot exist in the input tensor."""
        with self.assertRaisesRegex(ValueError, "out-of-range index 9"):
            DualBranchLCTRieszLSTMForecaster(
                input_dim=9,
                hidden_dim=8,
                lstm_hidden_dim=12,
                signal_feature_indices=[5, 6, 7, 9],
                num_layers=1,
                output_dim=1,
            )

    def test_dual_branch_smoke_training_writes_artifacts_and_index(self) -> None:
        """Train the dual-branch model briefly without touching real outputs."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "outputs"
            checkpoints_root = root / "checkpoints"
            config_path = root / "dual_branch_smoke.yaml"
            config = self._build_config(csv_path, outputs_root, checkpoints_root)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            paths = main(config_path)

            required_files = [
                paths.config_path,
                paths.training_log_path,
                paths.metrics_path,
                paths.prediction_results_path,
                paths.summary_path,
                paths.lct_parameters_path,
                paths.best_model_path,
                paths.latest_model_path,
                paths.figures_dir / "loss_curve.png",
                paths.figures_dir / "prediction_curve.png",
                paths.figures_dir / "residual_distribution.png",
                paths.figures_dir / "error_histogram.png",
                paths.figures_dir / "lct_parameter_analysis.png",
            ]
            for path in required_files:
                self.assertTrue(path.is_file(), "Missing artifact: {}".format(path))
                self.assertGreater(path.stat().st_size, 0)

            self.assertEqual(
                paths.experiment_dir.parent,
                outputs_root / "volatility_5" / "dual_branch_signal_features",
            )
            self.assertEqual(
                paths.checkpoint_dir.parent,
                checkpoints_root / "volatility_5" / "dual_branch_signal_features",
            )

            metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
            self.assertIn("rmse", metrics)
            self.assertIsNone(metrics["directional_accuracy"])

            index_path = outputs_root / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            dual_rows = [
                row for row in index_rows
                if row["run_dir"] == str(paths.experiment_dir)
            ]
            self.assertEqual(len(dual_rows), 1)
            self.assertEqual(
                dual_rows[0]["model_type"],
                "dual_branch_lct_riesz_lstm",
            )
            self.assertEqual(dual_rows[0]["target_type"], "volatility_5")
            self.assertEqual(dual_rows[0]["use_lct_riesz"], "True")

    @staticmethod
    def _build_config(
        csv_path: Path,
        outputs_root: Path,
        checkpoints_root: Path,
    ) -> dict:
        """Return a small dual-branch volatility configuration."""
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
                "type": "dual_branch_lct_riesz_lstm",
                "input_dim": 9,
                "hidden_dim": 8,
                "lstm_hidden_dim": 16,
                "num_layers": 1,
                "output_dim": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "use_lct_riesz": True,
                "signal_feature_indices": [5, 6, 7, 8],
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "seed": 42,
                "device": "cpu",
            },
            "experiment": {
                "name": "lstm_volatility_5_dual_branch_signal_features",
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
