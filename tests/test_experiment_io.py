"""Tests for reproducible experiment artifact persistence."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.utils.experiment_io import (
    EXPERIMENT_INDEX_FIELDS,
    append_experiment_index,
    append_training_log,
    classify_experiment_run,
    create_experiment_dir,
    save_checkpoint,
    save_config,
    save_experiment_summary,
    save_lct_parameters,
    save_metrics,
    save_prediction_results,
)


class ExperimentIOTest(unittest.TestCase):
    """Verify directory isolation and every required experiment artifact."""

    def setUp(self) -> None:
        """Create isolated output and checkpoint roots."""
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.outputs_root = root / "outputs"
        self.checkpoints_root = root / "checkpoints"
        self.paths = create_experiment_dir(
            outputs_root=self.outputs_root,
            checkpoints_root=self.checkpoints_root,
            timestamp=datetime(2026, 6, 12, 10, 30, 45),
        )

    def tearDown(self) -> None:
        """Remove all temporary experiment artifacts."""
        self.temp_dir.cleanup()

    def test_create_experiment_dir_builds_complete_unique_structure(self) -> None:
        """Create all run directories without overwriting a same-second run."""
        self.assertTrue(self.paths.experiment_dir.is_dir())
        self.assertTrue(self.paths.checkpoint_dir.is_dir())
        self.assertTrue(self.paths.figures_dir.is_dir())
        self.assertEqual(
            self.paths.experiment_dir.parent,
            self.outputs_root / "archive" / "unknown",
        )
        self.assertEqual(
            self.paths.checkpoint_dir.parent,
            self.checkpoints_root / "archive" / "unknown",
        )
        self.assertEqual(
            self.paths.experiment_dir.name,
            "20260612_103045",
        )

        second_paths = create_experiment_dir(
            outputs_root=self.outputs_root,
            checkpoints_root=self.checkpoints_root,
            timestamp=datetime(2026, 6, 12, 10, 30, 45),
        )
        self.assertEqual(
            second_paths.experiment_dir.name,
            "20260612_103045_01",
        )
        self.assertNotEqual(
            self.paths.experiment_dir,
            second_paths.experiment_dir,
        )

    def test_create_experiment_dir_appends_safe_experiment_name(self) -> None:
        """Append a sanitized experiment name without creating unsafe paths."""
        named_paths = create_experiment_dir(
            outputs_root=self.outputs_root,
            checkpoints_root=self.checkpoints_root,
            timestamp=datetime(2026, 6, 12, 10, 30, 46),
            prefix="experiment",
            experiment_name="LSTM Log Return 中文 !@# LCT/Riesz",
        )

        self.assertEqual(
            named_paths.experiment_dir.parent,
            self.outputs_root / "log_return" / "lct_riesz",
        )
        self.assertEqual(
            named_paths.experiment_dir.name,
            "20260612_103046",
        )
        self.assertNotIn(" ", named_paths.experiment_dir.name)
        self.assertNotIn("/", named_paths.experiment_dir.name)
        self.assertNotIn("\\", named_paths.experiment_dir.name)

    def test_create_experiment_dir_keeps_duplicate_suffix_with_name(self) -> None:
        """Avoid overwriting same-second named experiment directories."""
        first_paths = create_experiment_dir(
            outputs_root=self.outputs_root,
            checkpoints_root=self.checkpoints_root,
            timestamp=datetime(2026, 6, 12, 10, 30, 47),
            experiment_name="lstm_log_return_lct_riesz",
        )
        second_paths = create_experiment_dir(
            outputs_root=self.outputs_root,
            checkpoints_root=self.checkpoints_root,
            timestamp=datetime(2026, 6, 12, 10, 30, 47),
            experiment_name="lstm_log_return_lct_riesz",
        )
        third_paths = create_experiment_dir(
            outputs_root=self.outputs_root,
            checkpoints_root=self.checkpoints_root,
            timestamp=datetime(2026, 6, 12, 10, 30, 47),
            experiment_name="lstm_log_return_lct_riesz",
        )

        self.assertEqual(
            first_paths.experiment_dir.name,
            "20260612_103047",
        )
        self.assertEqual(
            second_paths.experiment_dir.name,
            "20260612_103047_01",
        )
        self.assertEqual(
            third_paths.experiment_dir.name,
            "20260612_103047_02",
        )

    def test_classify_experiment_run_covers_current_research_buckets(self) -> None:
        """Classify LCT, baseline, naive, and unknown runs into stable buckets."""
        volatility_lct = classify_experiment_run(
            experiment_name="lstm_volatility_5_lct_signal_features",
        )
        volatility_baseline = classify_experiment_run(
            experiment_name="lstm_volatility_5_baseline_features",
        )
        log_return = classify_experiment_run(
            config={
                "data": {"target_type": "log_return"},
                "model": {"use_lct_riesz": True},
            },
        )
        naive = classify_experiment_run(
            prefix="naive",
            experiment_name="naive_historical_volatility_5",
        )
        unknown = classify_experiment_run(experiment_name="close_smoke_test")

        self.assertEqual(volatility_lct.task_name, "volatility_5")
        self.assertEqual(volatility_lct.run_type, "lct_signal_features")
        self.assertEqual(volatility_baseline.task_name, "volatility_5")
        self.assertEqual(volatility_baseline.run_type, "baseline_features")
        self.assertEqual(log_return.task_name, "log_return")
        self.assertEqual(log_return.run_type, "lct_riesz")
        self.assertEqual(naive.task_name, "naive")
        self.assertEqual(naive.run_type, "historical_volatility_5")
        self.assertEqual(unknown.task_name, "archive")
        self.assertEqual(unknown.run_type, "unknown")

    def test_append_experiment_index_creates_and_appends_rows(self) -> None:
        """Create an experiment index and tolerate missing optional fields."""
        index_path = append_experiment_index(
            outputs_root=self.outputs_root,
            run_dir="outputs/run_a",
            checkpoint_dir="checkpoints/run_a",
            prefix="experiment",
            experiment_name="run_a",
            target_type="return",
            model_type="lct_riesz_lstm",
            use_lct_riesz=True,
            epochs=2,
            rmse=0.1,
        )
        append_experiment_index(
            outputs_root=self.outputs_root,
            run_dir="outputs/run_b",
            prefix="naive",
            model_type="naive",
            naive_rule="zero_return",
        )

        with index_path.open("r", newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0]), EXPERIMENT_INDEX_FIELDS)
        self.assertEqual(rows[0]["experiment_name"], "run_a")
        self.assertEqual(rows[0]["use_lct_riesz"], "True")
        self.assertEqual(rows[0]["rmse"], "0.1")
        self.assertEqual(rows[1]["model_type"], "naive")
        self.assertEqual(rows[1]["naive_rule"], "zero_return")
        self.assertEqual(rows[1]["checkpoint_dir"], "")
        self.assertTrue(rows[1]["created_at"])

    def test_save_config_supports_utf8_and_scientific_scalars(self) -> None:
        """Save readable configuration JSON without escaping Chinese text."""
        config = {
            "dataset": "沪深股票",
            "learning_rate": np.float32(0.001),
            "device": "cuda",
        }
        save_config(config, self.paths.config_path)

        content = self.paths.config_path.read_text(encoding="utf-8")
        self.assertIn("沪深股票", content)
        loaded = json.loads(content)
        self.assertAlmostEqual(loaded["learning_rate"], 0.001)

    def test_append_training_log_writes_and_appends_rows(self) -> None:
        """Write a CSV header once and append multiple epoch rows."""
        append_training_log(
            {
                "epoch": 1,
                "train_loss": 0.5,
                "val_loss": 0.6,
                "learning_rate": 0.001,
            },
            self.paths.training_log_path,
        )
        append_training_log(
            {
                "epoch": 2,
                "train_loss": 0.4,
                "val_loss": 0.5,
                "learning_rate": 0.0005,
                "elapsed_seconds": 2.5,
            },
            self.paths.training_log_path,
        )

        with self.paths.training_log_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["epoch"], "1")
        self.assertEqual(rows[1]["elapsed_seconds"], "2.5")

    def test_save_metrics_writes_json(self) -> None:
        """Save standard and directional regression metrics."""
        metrics = {
            "MAE": np.float64(0.1),
            "MSE": 0.02,
            "RMSE": 0.1414,
            "MAPE": 1.2,
            "R2": 0.95,
            "directional_accuracy": torch.tensor(0.75),
        }
        save_metrics(metrics, self.paths.metrics_path)

        loaded = json.loads(self.paths.metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(set(loaded), set(metrics))
        self.assertEqual(loaded["directional_accuracy"], 0.75)

    def test_save_prediction_results_writes_error_column(self) -> None:
        """Save aligned targets, predictions, and prediction-minus-target errors."""
        save_prediction_results(
            y_true=torch.tensor([1.0, 2.0, 3.0]),
            y_pred=np.array([1.1, 1.8, 3.2]),
            path=self.paths.prediction_results_path,
        )

        with self.paths.prediction_results_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(list(rows[0]), ["y_true", "y_pred", "error"])
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(float(rows[0]["error"]), 0.1)

    def test_save_lct_parameters_writes_readable_matrix(self) -> None:
        """Save every required LCT-Riesz parameter as structured text."""
        params = {
            "alpha": 1.0,
            "m": 1.0,
            "q": 0.0,
            "A": 0.0,
            "B": 1.0,
            "C": -1.0,
            "D": 0.0,
            "gamma": 1.0,
        }
        save_lct_parameters(params, self.paths.lct_parameters_path)

        content = self.paths.lct_parameters_path.read_text(encoding="utf-8")
        for field in params:
            self.assertIn(field, content)
        self.assertIn("determinant (A*D - B*C) = 1", content)

    def test_save_experiment_summary_supports_string_and_mapping(self) -> None:
        """Save both structured and free-form experiment summaries."""
        save_experiment_summary(
            {"model": "LCT-Riesz-LSTM", "结论": "测试通过"},
            self.paths.summary_path,
        )
        content = self.paths.summary_path.read_text(encoding="utf-8")
        self.assertIn("LCT-Riesz-LSTM", content)
        self.assertIn("测试通过", content)

        save_experiment_summary("Plain summary", self.paths.summary_path)
        self.assertEqual(
            self.paths.summary_path.read_text(encoding="utf-8"),
            "Plain summary\n",
        )

    def test_save_checkpoint_writes_loadable_pytorch_state(self) -> None:
        """Save a checkpoint using a small model without real training."""
        model = nn.Linear(3, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=4,
            metrics={"val_loss": np.float32(0.25)},
            path=self.paths.best_model_path,
        )

        checkpoint = torch.load(
            self.paths.best_model_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(
            set(checkpoint),
            {
                "model_state_dict",
                "optimizer_state_dict",
                "epoch",
                "metrics",
            },
        )
        self.assertEqual(checkpoint["epoch"], 4)
        self.assertAlmostEqual(checkpoint["metrics"]["val_loss"], 0.25)


if __name__ == "__main__":
    unittest.main()
