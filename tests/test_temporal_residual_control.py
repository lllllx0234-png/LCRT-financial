"""Tests for the parameter-matched causal temporal residual control."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from diagnose_residual_contribution import run_diagnostic
from evaluate import run_evaluation
from src.models.lct_riesz_1d import LearnableLCT1D, LearnableLCTRiesz1D
from src.models.temporal_residual_lstm import (
    ResidualAuxiliaryTemporalLSTMForecaster,
    SharedCausalTemporalConv1D,
)
from src.utils.experiment_io import classify_experiment_run
from train import build_model, main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "experiments" / "lstm" / "configs"
LCT_CONFIG_PATH = (
    CONFIG_ROOT / "lstm_volatility_5_residual_lct_signal_features.yaml"
)
NO_LCT_CONFIG_PATH = (
    CONFIG_ROOT / "lstm_volatility_5_residual_no_lct_signal_features.yaml"
)
TEMPORAL_CONFIG_PATH = (
    CONFIG_ROOT / "lstm_volatility_5_residual_temporal_signal_features.yaml"
)
MAIN_PREFIXES = ("input_projection.", "main_lstm.", "main_head.")
SHARED_PREFIXES = (
    *MAIN_PREFIXES,
    "spectral_projection.",
    "spectral_delta_head.",
    "residual_scale",
)


class TemporalResidualControlTest(unittest.TestCase):
    """Validate structure, causality, fairness, and pipeline integration."""

    def test_parameter_match_and_shared_initialization_are_exact(self) -> None:
        """Match LCT parameter count and all comparator-shared initialization."""
        lct_config = self._load_yaml(LCT_CONFIG_PATH)
        no_lct_config = self._load_yaml(NO_LCT_CONFIG_PATH)
        temporal_config = self._load_yaml(TEMPORAL_CONFIG_PATH)

        torch.manual_seed(42)
        lct_model = build_model(lct_config["model"])
        torch.manual_seed(42)
        no_lct_model = build_model(no_lct_config["model"])
        torch.manual_seed(42)
        temporal_model = build_model(temporal_config["model"])

        self.assertIsInstance(
            temporal_model,
            ResidualAuxiliaryTemporalLSTMForecaster,
        )
        self.assertFalse(temporal_model.use_lct_riesz)
        self.assertEqual(temporal_model.count_parameters(), 59067)
        self.assertEqual(lct_model.count_parameters(), 59067)
        self.assertEqual(no_lct_model.count_parameters(), 59011)
        self.assertEqual(
            sum(parameter.numel() for parameter in temporal_model.temporal_block.parameters()),
            56,
        )
        self.assertEqual(
            tuple(temporal_model.temporal_block.state_dict()),
            ("conv.weight",),
        )
        self.assertIsNone(temporal_model.export_lct_parameters())
        self.assertFalse(
            any(
                isinstance(module, (LearnableLCT1D, LearnableLCTRiesz1D))
                for module in temporal_model.modules()
            )
        )
        self.assertFalse(
            any(
                "lct" in name.lower() or "riesz" in name.lower()
                for name in temporal_model.state_dict()
            )
        )

        lct_shared = self._selected_state(lct_model, SHARED_PREFIXES)
        temporal_shared = self._selected_state(temporal_model, SHARED_PREFIXES)
        self.assertEqual(tuple(lct_shared), tuple(temporal_shared))
        for name in lct_shared:
            self.assertEqual(lct_shared[name].shape, temporal_shared[name].shape)
            self.assertTrue(
                torch.equal(lct_shared[name], temporal_shared[name]),
                "Shared parameter differs at fixed seed: {}".format(name),
            )

        no_lct_main = self._selected_state(no_lct_model, MAIN_PREFIXES)
        temporal_main = self._selected_state(temporal_model, MAIN_PREFIXES)
        self.assertEqual(tuple(no_lct_main), tuple(temporal_main))
        for name in no_lct_main:
            self.assertTrue(torch.equal(no_lct_main[name], temporal_main[name]))

    def test_temporal_block_is_strictly_causal_shared_and_length_preserving(self) -> None:
        """Test intermediate responses rather than pooled final predictions."""
        block = SharedCausalTemporalConv1D(channels=4, kernel_size=56)
        inputs = torch.randn(2, 4, 60)
        response = block.causal_response(inputs)
        enhanced = block(inputs)
        self.assertEqual(tuple(response.shape), (2, 4, 60))
        self.assertEqual(tuple(enhanced.shape), (2, 4, 60))
        self.assertEqual(block.receptive_field, 56)
        self.assertEqual(block.causal_left_padding, 55)
        self.assertTrue(block.shared_across_channels)
        self.assertIsNone(block.conv.bias)

        changed_future = inputs.clone()
        changed_future[:, :, 31:] += 1000.0
        changed_response = block.causal_response(changed_future)
        torch.testing.assert_close(
            response[:, :, :31],
            changed_response[:, :, :31],
            rtol=0.0,
            atol=0.0,
        )

        repeated_channel = torch.randn(2, 1, 60).expand(-1, 4, -1).clone()
        shared_response = block.causal_response(repeated_channel)
        for channel in range(1, 4):
            torch.testing.assert_close(
                shared_response[:, 0],
                shared_response[:, channel],
                rtol=0.0,
                atol=0.0,
            )

        with torch.no_grad():
            block.conv.weight.fill_(1.0)
        impulse = torch.zeros(1, 4, 60)
        impulse[:, :, 0] = 1.0
        impulse_response = block.causal_response(impulse)
        self.assertTrue(torch.equal(impulse_response[:, :, 55], torch.ones(1, 4)))
        self.assertTrue(torch.equal(impulse_response[:, :, 56], torch.zeros(1, 4)))

    def test_forward_components_are_finite_and_temporal_weights_receive_gradients(self) -> None:
        """Verify residual identities and gradient flow through all 56 weights."""
        config = self._load_yaml(TEMPORAL_CONFIG_PATH)
        model = build_model(config["model"])
        inputs = torch.randn(5, 60, 9)

        model.eval()
        output = model(inputs)
        components = model.forward_components(inputs)
        self.assertEqual(tuple(output.shape), (5, 1))
        self.assertEqual(
            set(components),
            {
                "main_pred",
                "auxiliary_delta",
                "residual_scale",
                "residual_correction",
                "final_pred",
            },
        )
        self.assertTrue(torch.equal(output, components["final_pred"]))
        self.assertTrue(
            torch.equal(
                components["residual_correction"],
                components["residual_scale"] * components["auxiliary_delta"],
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
        model.residual_scale.data.fill_(1.0)
        loss = model.forward_components(inputs)["final_pred"].square().mean()
        loss.backward()
        gradient = model.temporal_block.conv.weight.grad
        self.assertIsNotNone(gradient)
        self.assertEqual(gradient.numel(), 56)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.count_nonzero(gradient).item() == 56)
        self.assertIsNotNone(model.input_projection.weight.grad)
        self.assertIsNotNone(model.spectral_projection[0].weight.grad)
        self.assertIsNotNone(model.residual_scale.grad)

    def test_factory_formal_config_and_output_classification_are_fair(self) -> None:
        """Restrict formal differences to temporal identity and explicit block fields."""
        lct_config = self._load_yaml(LCT_CONFIG_PATH)
        temporal_config = self._load_yaml(TEMPORAL_CONFIG_PATH)

        self.assertEqual(temporal_config["data"], lct_config["data"])
        self.assertEqual(temporal_config["training"], lct_config["training"])
        self.assertFalse(temporal_config["data"].get("shuffle_train", False))
        self.assertEqual(
            temporal_config["model"]["type"],
            "residual_auxiliary_temporal_lstm",
        )
        self.assertFalse(temporal_config["model"]["use_lct_riesz"])

        temporal_fields = {
            "temporal_kernel_size": 56,
            "temporal_bias": False,
            "temporal_shared_across_channels": True,
            "temporal_causal_left_padding": 55,
            "temporal_activation": "gelu",
            "temporal_residual_connection": True,
        }
        for name, expected in temporal_fields.items():
            self.assertEqual(temporal_config["model"][name], expected)

        lct_model_config = copy.deepcopy(lct_config["model"])
        temporal_model_config = copy.deepcopy(temporal_config["model"])
        for config in (lct_model_config, temporal_model_config):
            config.pop("type")
            config.pop("use_lct_riesz")
        for name in temporal_fields:
            temporal_model_config.pop(name)
        self.assertEqual(temporal_model_config, lct_model_config)

        lct_experiment = copy.deepcopy(lct_config["experiment"])
        temporal_experiment = copy.deepcopy(temporal_config["experiment"])
        lct_experiment.pop("name")
        temporal_experiment.pop("name")
        self.assertEqual(temporal_experiment, lct_experiment)

        classification = classify_experiment_run(
            experiment_name=temporal_config["experiment"]["name"],
            config=temporal_config,
            config_path=TEMPORAL_CONFIG_PATH,
        )
        self.assertEqual(classification.task_name, "volatility_5")
        self.assertEqual(
            classification.run_type,
            "residual_temporal_signal_features",
        )

        invalid_config = copy.deepcopy(temporal_config["model"])
        invalid_config["use_lct_riesz"] = True
        with self.assertRaisesRegex(ValueError, "requires use_lct_riesz=false"):
            build_model(invalid_config)

    def test_existing_lct_checkpoint_remains_strictly_loadable(self) -> None:
        """Protect existing residual LCT state keys and checkpoint compatibility."""
        checkpoint_path = (
            PROJECT_ROOT
            / "experiments"
            / "lstm"
            / "checkpoints"
            / "volatility_5"
            / "residual_lct_signal_features"
            / "20260625_163554"
            / "best_model.pth"
        )
        if not checkpoint_path.is_file():
            self.skipTest("Formal residual LCT checkpoint is unavailable locally.")
        lct_config = self._load_yaml(LCT_CONFIG_PATH)
        model = build_model(lct_config["model"])
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

    def test_train_evaluate_and_full_diagnostic_smoke_stay_temporary(self) -> None:
        """Exercise factory, evaluation, and every diagnostic in a temporary root."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "data" / "sample.csv"
            csv_path.parent.mkdir(parents=True)
            self._write_ohlcv_csv(csv_path)

            outputs_root = root / "outputs"
            checkpoints_root = root / "checkpoints"
            config_path = root / "temporal_residual_smoke.yaml"
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
                / "residual_temporal_signal_features",
            )
            summary = training_paths.summary_path.read_text(encoding="utf-8")
            self.assertIn("model_type: residual_auxiliary_temporal_lstm", summary)
            self.assertIn("use_lct_riesz: False", summary)

            evaluation_paths = run_evaluation(
                config_path,
                training_paths.best_model_path,
            )
            self.assertTrue(evaluation_paths.metrics_path.is_file())
            self.assertFalse(evaluation_paths.lct_parameters_path.exists())
            self.assertEqual(
                evaluation_paths.experiment_dir.parent,
                outputs_root
                / "volatility_5"
                / "residual_temporal_signal_features",
            )

            diagnostic_dir = root / "diagnostics" / "temporal"
            resolved_diagnostic = run_diagnostic(
                config_path=training_paths.config_path,
                checkpoint_path=training_paths.best_model_path,
                original_metrics_path=training_paths.metrics_path,
                original_predictions_path=training_paths.prediction_results_path,
                output_dir=diagnostic_dir,
                num_permutations=4,
                seed=42,
            )
            self.assertEqual(resolved_diagnostic, diagnostic_dir)
            required_diagnostics = (
                "residual_ablation_metrics.json",
                "residual_contribution_summary.json",
                "residual_components.csv",
                "residual_regime_metrics.csv",
                "permutation_metrics.csv",
                "permutation_summary.json",
            )
            for name in required_diagnostics:
                self.assertTrue((diagnostic_dir / name).is_file(), name)

            ablation = json.loads(
                (diagnostic_dir / "residual_ablation_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                ablation["model"],
                {
                    "type": "residual_auxiliary_temporal_lstm",
                    "auxiliary_type": "causal_temporal_conv",
                    "use_lct_riesz": False,
                    "kernel_size": 56,
                },
            )
            self.assertTrue(ablation["verification"]["matched"])
            contribution = json.loads(
                (diagnostic_dir / "residual_contribution_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("auxiliary_delta", contribution)
            self.assertNotIn("spectral_delta", contribution)
            self.assertEqual(contribution["source"]["kernel_size"], 56)
            with (diagnostic_dir / "residual_components.csv").open(
                "r", newline="", encoding="utf-8-sig"
            ) as file:
                columns = next(csv.reader(file))
            self.assertIn("auxiliary_delta", columns)
            self.assertNotIn("spectral_delta", columns)

            for path in (
                training_paths.experiment_dir,
                training_paths.checkpoint_dir,
                evaluation_paths.experiment_dir,
                diagnostic_dir,
            ):
                self.assertTrue(path.resolve().is_relative_to(root.resolve()))

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """Load a YAML mapping used by one control test."""
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    @staticmethod
    def _selected_state(
        model: torch.nn.Module,
        prefixes: tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        """Return cloned state entries whose names begin with shared prefixes."""
        return {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name.startswith(prefixes)
        }

    @staticmethod
    def _write_ohlcv_csv(path: Path) -> None:
        """Write deterministic positive OHLCV data for temporary smoke tests."""
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
        """Return a one-epoch temporal residual config for temporary smoke tests."""
        config = yaml.safe_load(TEMPORAL_CONFIG_PATH.read_text(encoding="utf-8"))
        config["data"]["csv_path"] = str(csv_path)
        config["data"]["sequence_length"] = 10
        config["data"]["batch_size"] = 4
        config["model"]["hidden_dim"] = 8
        config["model"]["lstm_hidden_dim"] = 16
        config["model"]["num_layers"] = 1
        config["model"]["dropout"] = 0.0
        config["training"]["epochs"] = 1
        config["training"]["device"] = "cpu"
        config["experiment"]["outputs_root"] = str(outputs_root)
        config["experiment"]["checkpoints_root"] = str(checkpoints_root)
        return config


if __name__ == "__main__":
    unittest.main()
