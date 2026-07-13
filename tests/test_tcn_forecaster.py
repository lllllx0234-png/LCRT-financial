"""Tests for the plain TCN financial forecaster."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.models.tcn_forecaster import CausalConv1d, TCNForecaster, TemporalBlock


class TCNForecasterTest(unittest.TestCase):
    """Validate plain TCN shape, gradients, causality, and interfaces."""

    def setUp(self) -> None:
        """Create deterministic signal-feature-shaped input data."""
        torch.manual_seed(42)
        self.batch_size = 4
        self.sequence_length = 60
        self.input_dim = 9
        self.x = torch.randn(self.batch_size, self.sequence_length, self.input_dim)
        self.model = TCNForecaster(
            input_dim=self.input_dim,
            channels=[64, 64, 64],
            kernel_size=3,
            dropout=0.1,
            output_dim=1,
        )

    def test_forward_output_shape(self) -> None:
        """Forward returns one regression output per sequence."""
        output = self.model(self.x)
        self.assertEqual(tuple(output.shape), (4, 1))

    def test_forward_features_output_shape(self) -> None:
        """Forward features keeps full temporal resolution."""
        features = self.model.forward_features(self.x)
        self.assertEqual(tuple(features.shape), (4, 64, 60))

    def test_backward_produces_finite_gradients(self) -> None:
        """A scalar regression loss propagates finite gradients."""
        prediction = self.model(self.x)
        loss = prediction.square().mean()
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertGreater(len(gradients), 0)
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())

    def test_sequence_length_is_preserved(self) -> None:
        """Each temporal block and forward_features preserve input length."""
        model = TCNForecaster(
            input_dim=self.input_dim,
            channels=[8, 16, 16],
            kernel_size=3,
            dropout=0.0,
            output_dim=1,
        )
        for sequence_length in (16, 31, 60):
            with self.subTest(sequence_length=sequence_length):
                x = torch.randn(2, sequence_length, self.input_dim)
                features = x.transpose(1, 2).contiguous()
                for block in model.temporal_network:
                    features = block(features)
                    self.assertEqual(features.shape[-1], sequence_length)
                output_features = model.forward_features(x)
                self.assertEqual(output_features.shape[-1], sequence_length)

    def test_causal_features_do_not_use_future_inputs(self) -> None:
        """Changes after a future index do not alter earlier features."""
        model = TCNForecaster(
            input_dim=self.input_dim,
            channels=[16, 16, 16],
            kernel_size=3,
            dropout=0.0,
            output_dim=1,
        )
        model.eval()
        x1 = torch.randn(2, self.sequence_length, self.input_dim)
        x2 = x1.clone()
        future_index = 40
        x2[:, future_index:, :] = torch.randn_like(x2[:, future_index:, :]) * 100.0

        with torch.no_grad():
            features1 = model.forward_features(x1)
            features2 = model.forward_features(x2)

        self.assertTrue(
            torch.allclose(
                features1[:, :, :future_index],
                features2[:, :, :future_index],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_temporal_block_residual_channel_mapping(self) -> None:
        """Residual branch handles equal and changed channel counts."""
        same_channels = TemporalBlock(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            dilation=1,
            dropout=0.0,
        )
        changed_channels = TemporalBlock(
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            dilation=2,
            dropout=0.0,
        )
        x = torch.randn(2, 8, 20)

        same_output = same_channels(x)
        changed_output = changed_channels(x)

        self.assertEqual(tuple(same_output.shape), (2, 8, 20))
        self.assertEqual(tuple(changed_output.shape), (2, 12, 20))
        self.assertIsInstance(same_channels.downsample, nn.Identity)
        self.assertIsInstance(changed_channels.downsample, nn.Conv1d)

    def test_interface_compatibility(self) -> None:
        """Plain TCN exposes training-framework-compatible helpers."""
        self.assertIs(self.model.use_lct_riesz, False)
        self.assertIsNone(self.model.export_lct_parameters())
        self.assertIsInstance(self.model.count_parameters(), int)
        self.assertGreater(self.model.count_parameters(), 0)

    def test_invalid_model_configuration_raises(self) -> None:
        """Reject invalid TCN constructor arguments."""
        invalid_configs = [
            {"input_dim": 0, "channels": [8], "kernel_size": 3, "dropout": 0.0},
            {"input_dim": 9, "channels": [], "kernel_size": 3, "dropout": 0.0},
            {"input_dim": 9, "channels": [8, 0], "kernel_size": 3, "dropout": 0.0},
            {"input_dim": 9, "channels": [8], "kernel_size": 0, "dropout": 0.0},
            {"input_dim": 9, "channels": [8], "kernel_size": 3, "dropout": -0.1},
            {"input_dim": 9, "channels": [8], "kernel_size": 3, "dropout": 1.0},
            {
                "input_dim": 9,
                "channels": [8],
                "kernel_size": 3,
                "dropout": 0.0,
                "output_dim": 0,
            },
        ]
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises((TypeError, ValueError)):
                    TCNForecaster(**config)

    def test_invalid_forward_inputs_raise(self) -> None:
        """Reject malformed or non-floating input tensors."""
        model = TCNForecaster(
            input_dim=self.input_dim,
            channels=[8],
            kernel_size=3,
            dropout=0.0,
            output_dim=1,
        )
        invalid_inputs = [
            torch.randn(4, self.input_dim),
            torch.randn(4, 10, self.input_dim + 1),
            torch.randn(4, 0, self.input_dim),
            torch.ones(4, 10, self.input_dim, dtype=torch.long),
        ]
        for x in invalid_inputs:
            with self.subTest(shape=tuple(x.shape), dtype=x.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    model(x)


class CausalConv1dTest(unittest.TestCase):
    """Validate causal convolution parameter checks."""

    def test_invalid_causal_conv_configuration_raises(self) -> None:
        """Reject non-positive causal convolution dimensions."""
        invalid_configs = [
            {"in_channels": 0, "out_channels": 4, "kernel_size": 3, "dilation": 1},
            {"in_channels": 4, "out_channels": 0, "kernel_size": 3, "dilation": 1},
            {"in_channels": 4, "out_channels": 4, "kernel_size": 0, "dilation": 1},
            {"in_channels": 4, "out_channels": 4, "kernel_size": 3, "dilation": 0},
        ]
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises((TypeError, ValueError)):
                    CausalConv1d(**config)


if __name__ == "__main__":
    unittest.main()
