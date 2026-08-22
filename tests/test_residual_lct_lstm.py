"""Tests for the residual auxiliary LCT-Riesz LSTM forecaster."""

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

from src.models.residual_lct_lstm import ResidualAuxiliaryLCTRieszLSTMForecaster
from evaluate import run_evaluation
from train import build_model, main


class ResidualAuxiliaryLCTRieszLSTMTest(unittest.TestCase):
    """Validate residual LCT-Riesz correction and training integration."""

    def test_forward_components_backward_and_parameter_export(self) -> None:
        """Validate component identities, compatibility, and gradients."""
        model = ResidualAuxiliaryLCTRieszLSTMForecaster(
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
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        state_keys_before = tuple(model.state_dict())

        output = model(x)
        components = model.forward_components(x)
        self.assertEqual(tuple(output.shape), (4, 1))
        self.assertTrue(torch.equal(output, components["final_pred"]))
        self.assertEqual(
            set(components),
            {
                "main_pred",
                "spectral_delta",
                "residual_scale",
                "residual_correction",
                "final_pred",
            },
        )
        for name in (
            "main_pred",
            "spectral_delta",
            "residual_correction",
            "final_pred",
        ):
            self.assertEqual(tuple(components[name].shape), (4, 1))
            self.assertTrue(torch.isfinite(components[name]).all())
        self.assertEqual(tuple(components["residual_scale"].shape), ())
        self.assertTrue(torch.isfinite(components["residual_scale"]).all())
        self.assertTrue(
            torch.equal(
                components["residual_correction"],
                components["residual_scale"] * components["spectral_delta"],
            )
        )
        self.assertTrue(
            torch.equal(
                components["final_pred"],
                components["main_pred"] + components["residual_correction"],
            )
        )
        self.assertEqual(tuple(model.state_dict()), state_keys_before)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter.detach(), parameters_before[name]))

        loss = components["final_pred"].mean()
        loss.backward()
        self.assertIsNotNone(model.input_projection.weight.grad)
        self.assertIsNotNone(model.residual_scale.grad)

        params = model.export_lct_parameters()
        self.assertIsNotNone(params)
        self.assertEqual(
            set(params or {}),
            {
                "alpha",
                "m",
                "q",
                "A",
                "B",
                "C",
                "D",
                "gamma",
                "residual_scale",
            },
        )
        self.assertAlmostEqual((params or {})["residual_scale"], 0.0)

    def test_formal_checkpoint_strictly_loads_with_component_interface(self) -> None:
        """Strictly load the existing formal residual checkpoint when present."""
        project_root = Path(__file__).resolve().parents[1]
        run_dir = (
            project_root
            / "experiments"
            / "lstm"
            / "outputs"
            / "volatility_5"
            / "residual_lct_signal_features"
            / "20260625_163554"
        )
        checkpoint_path = (
            project_root
            / "experiments"
            / "lstm"
            / "checkpoints"
            / "volatility_5"
            / "residual_lct_signal_features"
            / "20260625_163554"
            / "best_model.pth"
        )
        if not checkpoint_path.is_file():
            self.skipTest("Formal residual checkpoint is not available locally.")

        config = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
        model = build_model(config["model"])
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        incompatible = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(checkpoint["epoch"], 5)
        self.assertEqual(model.count_parameters(), 59067)

    def test_out_of_range_signal_feature_indices_raise_value_error(self) -> None:
        """Reject signal feature indices that exceed input_dim."""
        with self.assertRaisesRegex(ValueError, "out-of-range index 9"):
            ResidualAuxiliaryLCTRieszLSTMForecaster(
                input_dim=9,
                hidden_dim=8,
                lstm_hidden_dim=12,
                signal_feature_indices=[5, 6, 7, 9],
                num_layers=1,
                output_dim=1,
            )

    def test_residual_smoke_training_writes_artifacts_and_index(self) -> None:
        """Train residual auxiliary LCT briefly in a temporary workspace."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "raw" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "outputs"
            checkpoints_root = root / "checkpoints"
            config_path = root / "residual_smoke.yaml"
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
                outputs_root / "volatility_5" / "residual_lct_signal_features",
            )
            self.assertEqual(
                paths.checkpoint_dir.parent,
                checkpoints_root / "volatility_5" / "residual_lct_signal_features",
            )

            metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
            self.assertIn("rmse", metrics)
            self.assertIsNone(metrics["directional_accuracy"])

            lct_parameters = paths.lct_parameters_path.read_text(encoding="utf-8")
            self.assertIn("residual_scale", lct_parameters)

            index_path = outputs_root / "experiment_index.csv"
            self.assertTrue(index_path.is_file())
            with index_path.open("r", newline="", encoding="utf-8-sig") as file:
                index_rows = list(csv.DictReader(file))
            residual_rows = [
                row for row in index_rows
                if row["run_dir"] == str(paths.experiment_dir)
            ]
            self.assertEqual(len(residual_rows), 1)
            self.assertEqual(
                residual_rows[0]["model_type"],
                "residual_auxiliary_lct_riesz_lstm",
            )
            self.assertEqual(residual_rows[0]["target_type"], "volatility_5")
            self.assertEqual(residual_rows[0]["use_lct_riesz"], "True")

            evaluation_paths = run_evaluation(
                config_path,
                paths.best_model_path,
            )
            self.assertTrue(evaluation_paths.metrics_path.is_file())
            evaluation_metrics = json.loads(
                evaluation_paths.metrics_path.read_text(encoding="utf-8")
            )
            self.assertIn("rmse", evaluation_metrics)
            self.assertIsNone(evaluation_metrics["directional_accuracy"])

    @staticmethod
    def _build_config(
        csv_path: Path,
        outputs_root: Path,
        checkpoints_root: Path,
    ) -> dict:
        """Return a small residual auxiliary volatility configuration."""
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
                "type": "residual_auxiliary_lct_riesz_lstm",
                "input_dim": 9,
                "hidden_dim": 8,
                "lstm_hidden_dim": 16,
                "num_layers": 1,
                "output_dim": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "use_lct_riesz": True,
                "signal_feature_indices": [5, 6, 7, 8],
                "residual_scale_init": 0.0,
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "seed": 42,
                "device": "cpu",
            },
            "experiment": {
                "name": "lstm_volatility_5_residual_lct_signal_features",
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
