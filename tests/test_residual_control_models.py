"""Tests for residual auxiliary control models without LCT-Riesz."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from diagnose_residual_contribution import collect_residual_components
from evaluate import run_evaluation
from src.models.lct_riesz_1d import LearnableLCTRiesz1D
from src.models.residual_lct_lstm import (
    ResidualAuxiliaryLCTRieszLSTMForecaster,
)
from src.utils.experiment_io import classify_experiment_run
from train import build_model, main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LCT_CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "lstm"
    / "configs"
    / "lstm_volatility_5_residual_lct_signal_features.yaml"
)
CONTROL_CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "lstm"
    / "configs"
    / "lstm_volatility_5_residual_no_lct_signal_features.yaml"
)
MAIN_PARAMETER_PREFIXES = ("input_projection.", "main_lstm.", "main_head.")


class ResidualControlModelsTest(unittest.TestCase):
    """Validate the parameter-fair No-LCT residual auxiliary control."""

    def test_no_lct_structure_parameter_counts_and_fair_initialization(self) -> None:
        """Remove only LCT-Riesz while preserving identical main initialization."""
        lct_config = self._load_yaml(LCT_CONFIG_PATH)
        control_config = self._load_yaml(CONTROL_CONFIG_PATH)

        torch.manual_seed(42)
        lct_model = build_model(lct_config["model"])
        torch.manual_seed(42)
        control_model = build_model(control_config["model"])

        self.assertIsInstance(
            control_model,
            ResidualAuxiliaryLCTRieszLSTMForecaster,
        )
        self.assertFalse(control_model.use_lct_riesz)
        self.assertIsInstance(control_model.lct_riesz, nn.Identity)
        self.assertFalse(
            any(
                isinstance(module, LearnableLCTRiesz1D)
                for module in control_model.modules()
            )
        )
        self.assertFalse(
            any(name.startswith("lct_riesz.") for name in control_model.state_dict())
        )
        self.assertIsNone(control_model.export_lct_parameters())
        self.assertEqual(control_model.count_parameters(), 59011)
        self.assertEqual(lct_model.count_parameters(), 59067)
        self.assertEqual(
            lct_model.count_parameters() - control_model.count_parameters(),
            56,
        )

        lct_main = self._main_parameters(lct_model)
        control_main = self._main_parameters(control_model)
        self.assertEqual(tuple(lct_main), tuple(control_main))
        for name in lct_main:
            self.assertEqual(lct_main[name].shape, control_main[name].shape)
            self.assertTrue(
                torch.equal(lct_main[name], control_main[name]),
                "Main parameter differs at fixed seed: {}".format(name),
            )

    def test_no_lct_forward_components_are_finite_and_differentiable(self) -> None:
        """Preserve residual component formulas, output shape, and gradients."""
        config = self._load_yaml(CONTROL_CONFIG_PATH)
        model = build_model(config["model"])
        inputs = torch.randn(5, 60, 9)

        model.eval()
        output = model(inputs)
        components = model.forward_components(inputs)
        self.assertEqual(tuple(output.shape), (5, 1))
        self.assertTrue(torch.equal(output, components["final_pred"]))
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
        for value in components.values():
            self.assertTrue(torch.isfinite(value).all())

        model.train()
        loss = model.forward_components(inputs)["final_pred"].square().mean()
        loss.backward()
        self.assertIsNotNone(model.input_projection.weight.grad)
        self.assertIsNotNone(model.spectral_projection[0].weight.grad)
        self.assertIsNotNone(model.residual_scale.grad)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_factory_rejects_lct_for_no_lct_model_type(self) -> None:
        """Prevent the control label from silently constructing an LCT model."""
        model_config = copy.deepcopy(
            self._load_yaml(CONTROL_CONFIG_PATH)["model"]
        )
        model_config["use_lct_riesz"] = True
        with self.assertRaisesRegex(ValueError, "requires use_lct_riesz=false"):
            build_model(model_config)

    def test_formal_control_config_is_fair_and_uniquely_classified(self) -> None:
        """Keep data and training fixed while isolating model identity and outputs."""
        lct_config = self._load_yaml(LCT_CONFIG_PATH)
        control_config = self._load_yaml(CONTROL_CONFIG_PATH)

        self.assertEqual(control_config["data"], lct_config["data"])
        self.assertEqual(control_config["training"], lct_config["training"])
        self.assertFalse(control_config["data"].get("shuffle_train", False))
        self.assertEqual(
            control_config["model"]["type"],
            "residual_auxiliary_signal_lstm",
        )
        self.assertFalse(control_config["model"]["use_lct_riesz"])

        lct_model_config = copy.deepcopy(lct_config["model"])
        control_model_config = copy.deepcopy(control_config["model"])
        lct_model_config.pop("type")
        control_model_config.pop("type")
        lct_model_config.pop("use_lct_riesz")
        control_model_config.pop("use_lct_riesz")
        self.assertEqual(control_model_config, lct_model_config)

        lct_experiment = copy.deepcopy(lct_config["experiment"])
        control_experiment = copy.deepcopy(control_config["experiment"])
        lct_experiment.pop("name")
        control_experiment.pop("name")
        self.assertEqual(control_experiment, lct_experiment)

        classification = classify_experiment_run(
            experiment_name=control_config["experiment"]["name"],
            config=control_config,
            config_path=CONTROL_CONFIG_PATH,
        )
        self.assertEqual(classification.task_name, "volatility_5")
        self.assertEqual(
            classification.run_type,
            "residual_no_lct_signal_features",
        )

    def test_no_lct_diagnostic_component_collection(self) -> None:
        """Use the common forward_components interface in diagnostic inference."""
        config = self._load_yaml(CONTROL_CONFIG_PATH)
        model = build_model(config["model"])
        inputs = torch.randn(7, 12, 9)
        targets = torch.randn(7, 1)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=3)

        components = collect_residual_components(
            model,
            loader,
            torch.device("cpu"),
        )
        self.assertEqual(components["target"].shape, (7,))
        self.assertEqual(components["final_pred"].shape, (7,))
        self.assertTrue(
            np.allclose(
                components["final_pred"],
                components["main_pred"] + components["residual_correction"],
            )
        )
        for name, values in components.items():
            self.assertTrue(np.isfinite(values).all(), name)

    def test_no_lct_train_and_evaluate_smoke_use_temporary_outputs(self) -> None:
        """Train and evaluate the factory control only inside a temporary root."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "outputs"
            checkpoints_root = root / "checkpoints"
            config_path = root / "residual_no_lct_smoke.yaml"
            config = self._smoke_config(
                csv_path,
                outputs_root,
                checkpoints_root,
            )
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            training_paths = main(config_path)
            self.assertTrue(training_paths.best_model_path.is_file())
            self.assertTrue(training_paths.metrics_path.is_file())
            self.assertFalse(training_paths.lct_parameters_path.exists())
            self.assertEqual(
                training_paths.experiment_dir.parent,
                outputs_root
                / "volatility_5"
                / "residual_no_lct_signal_features",
            )
            self.assertEqual(
                training_paths.checkpoint_dir.parent,
                checkpoints_root
                / "volatility_5"
                / "residual_no_lct_signal_features",
            )
            summary = training_paths.summary_path.read_text(encoding="utf-8")
            self.assertIn("model_type: residual_auxiliary_signal_lstm", summary)
            self.assertIn("use_lct_riesz: False", summary)

            evaluation_paths = run_evaluation(
                config_path,
                training_paths.best_model_path,
            )
            self.assertTrue(evaluation_paths.metrics_path.is_file())
            self.assertFalse(evaluation_paths.lct_parameters_path.exists())
            for path in (
                training_paths.experiment_dir,
                training_paths.checkpoint_dir,
                evaluation_paths.experiment_dir,
            ):
                self.assertTrue(path.resolve().is_relative_to(root.resolve()))

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """Load one test configuration as a mapping."""
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    @staticmethod
    def _main_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
        """Return detached main-branch parameters under their shared names."""
        return {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name.startswith(MAIN_PARAMETER_PREFIXES)
        }

    @staticmethod
    def _write_ohlcv_csv(path: Path) -> None:
        """Write deterministic OHLCV data for temporary smoke training."""
        rows = 120
        index = np.arange(rows, dtype=np.float64)
        close = 100.0 + 0.05 * index + np.sin(index / 6.0)
        pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=rows, freq="D"),
                "Open": close - 0.2,
                "High": close + 0.5,
                "Low": close - 0.6,
                "Close": close,
                "Volume": 1000.0 + 2.0 * index,
            }
        ).to_csv(path, index=False)

    @staticmethod
    def _smoke_config(
        csv_path: Path,
        outputs_root: Path,
        checkpoints_root: Path,
    ) -> dict:
        """Return a two-epoch No-LCT control config for temporary smoke tests."""
        features = [
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
                    *features,
                ],
                "derived_features": features,
                "target_type": "volatility_5",
                "sequence_length": 10,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "batch_size": 4,
                "num_workers": 0,
            },
            "model": {
                "type": "residual_auxiliary_signal_lstm",
                "input_dim": 9,
                "hidden_dim": 8,
                "lstm_hidden_dim": 16,
                "num_layers": 1,
                "output_dim": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "use_lct_riesz": False,
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
                "name": "lstm_volatility_5_residual_no_lct_signal_features",
                "save_outputs": True,
                "outputs_root": str(outputs_root),
                "checkpoints_root": str(checkpoints_root),
            },
        }


if __name__ == "__main__":
    unittest.main()
