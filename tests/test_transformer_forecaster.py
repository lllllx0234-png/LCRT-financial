"""Tests for the plain Transformer financial forecaster."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.models.transformer_forecaster import (
    LearnablePositionalEncoding,
    TransformerForecaster,
)


def _formal_transformer() -> TransformerForecaster:
    """Create the formal Plain Transformer comparison architecture."""
    return TransformerForecaster(
        input_dim=9,
        d_model=56,
        nhead=4,
        num_layers=2,
        dim_feedforward=112,
        dropout=0.1,
        max_sequence_length=60,
        output_dim=1,
        causal_attention=True,
    )


class TransformerForecasterTest(unittest.TestCase):
    """Validate shape, gradients, causality, and public interfaces."""

    def setUp(self) -> None:
        """Create deterministic signal-feature-shaped inputs."""
        torch.manual_seed(42)
        self.batch_size = 4
        self.sequence_length = 60
        self.input_dim = 9
        self.x = torch.randn(self.batch_size, self.sequence_length, self.input_dim)
        self.model = _formal_transformer()

    def test_forward_output_shape(self) -> None:
        """Forward returns one regression output per sequence."""
        output = self.model(self.x)
        self.assertEqual(tuple(output.shape), (4, 1))

    def test_forward_features_output_shape(self) -> None:
        """Forward features keeps the full temporal dimension."""
        features = self.model.forward_features(self.x)
        self.assertEqual(tuple(features.shape), (4, 60, 56))

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

        self.assertIsNotNone(self.model.input_projection.weight.grad)
        self.assertTrue(
            any(
                "self_attn" in name
                and parameter.requires_grad
                and parameter.grad is not None
                for name, parameter in self.model.named_parameters()
            )
        )
        self.assertIsNotNone(self.model.output_layer.weight.grad)

    def test_formal_parameter_count(self) -> None:
        """Formal architecture has the expected trainable parameter count."""
        parameter_count = self.model.count_parameters()
        direct_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

        self.assertIsInstance(parameter_count, int)
        self.assertGreater(parameter_count, 0)
        self.assertEqual(parameter_count, direct_count)
        self.assertEqual(parameter_count, 55497)

    def test_interface_compatibility(self) -> None:
        """Plain Transformer exposes training-framework-compatible helpers."""
        self.assertIs(self.model.use_lct_riesz, False)
        self.assertIsNone(self.model.export_lct_parameters())
        self.assertIs(self.model.causal_attention, True)

    def test_learnable_positional_encoding(self) -> None:
        """Position embeddings are trainable and handle valid lengths."""
        encoding = self.model.positional_encoding
        self.assertIsInstance(encoding.position_embedding, nn.Parameter)
        self.assertTrue(encoding.position_embedding.requires_grad)
        self.assertEqual(tuple(encoding.position_embedding.shape), (1, 60, 56))

        x = torch.randn(2, 31, 56)
        output = encoding(x)
        self.assertEqual(tuple(output.shape), (2, 31, 56))
        output.square().mean().backward()
        self.assertIsNotNone(encoding.position_embedding.grad)
        self.assertTrue(torch.isfinite(encoding.position_embedding.grad).all())

        short_output = encoding(torch.randn(2, 16, 56))
        self.assertEqual(tuple(short_output.shape), (2, 16, 56))
        with self.assertRaises(ValueError):
            encoding(torch.randn(2, 61, 56))

    def test_causal_features_do_not_use_future_inputs(self) -> None:
        """Changes after a future index do not alter earlier features."""
        model = _formal_transformer()
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
                features1[:, :future_index, :],
                features2[:, :future_index, :],
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_non_causal_mode_runs_without_causal_mask(self) -> None:
        """Non-causal mode forwards normally and does not build a mask."""
        model = TransformerForecaster(
            input_dim=9,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            max_sequence_length=60,
            output_dim=1,
            causal_attention=False,
        )
        x = torch.randn(2, 20, 9)

        self.assertIsNone(model._build_causal_mask(20, x.device))
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_different_sequence_lengths(self) -> None:
        """Forward paths support valid sequence lengths up to the configured max."""
        model = _formal_transformer()
        for sequence_length in (1, 16, 31, 60):
            with self.subTest(sequence_length=sequence_length):
                x = torch.randn(2, sequence_length, self.input_dim)
                features = model.forward_features(x)
                output = model(x)
                self.assertEqual(tuple(features.shape), (2, sequence_length, 56))
                self.assertEqual(tuple(output.shape), (2, 1))

    def test_batch_independence_in_eval_mode(self) -> None:
        """A sample prediction is unchanged when batched with another sample."""
        model = _formal_transformer()
        model.eval()
        sample = torch.randn(1, self.sequence_length, self.input_dim)
        other_sample = torch.randn(1, self.sequence_length, self.input_dim)
        batch = torch.cat([sample, other_sample], dim=0)

        with torch.no_grad():
            single_prediction = model(sample)
            batched_prediction = model(batch)[0:1]

        self.assertTrue(
            torch.allclose(
                single_prediction,
                batched_prediction,
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_invalid_model_configuration_raises(self) -> None:
        """Reject invalid Transformer constructor arguments."""
        valid_config = {
            "input_dim": 9,
            "d_model": 56,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 112,
            "dropout": 0.1,
            "max_sequence_length": 60,
            "output_dim": 1,
            "causal_attention": True,
        }
        invalid_overrides = [
            {"input_dim": 0},
            {"d_model": 0},
            {"nhead": 0},
            {"num_layers": 0},
            {"dim_feedforward": 0},
            {"max_sequence_length": 0},
            {"output_dim": 0},
            {"dropout": -0.1},
            {"dropout": 1.0},
            {"d_model": 55, "nhead": 4},
            {"causal_attention": "yes"},
        ]
        for override in invalid_overrides:
            config = {**valid_config, **override}
            with self.subTest(config=override):
                with self.assertRaises((TypeError, ValueError)):
                    TransformerForecaster(**config)

    def test_invalid_forward_inputs_raise(self) -> None:
        """Reject malformed or non-floating input tensors."""
        invalid_inputs = [
            "not a tensor",
            torch.randn(4, self.input_dim),
            torch.randn(0, 10, self.input_dim),
            torch.randn(4, 0, self.input_dim),
            torch.randn(4, 61, self.input_dim),
            torch.randn(4, 10, self.input_dim + 1),
            torch.ones(4, 10, self.input_dim, dtype=torch.long),
        ]
        for x in invalid_inputs:
            with self.subTest(value=repr(x)[:80]):
                with self.assertRaises((TypeError, ValueError)):
                    self.model(x)  # type: ignore[arg-type]


class LearnablePositionalEncodingTest(unittest.TestCase):
    """Validate standalone positional encoding configuration checks."""

    def test_invalid_configuration_raises(self) -> None:
        """Reject invalid positional encoding constructor arguments."""
        invalid_configs = [
            {"d_model": 0, "max_sequence_length": 60, "dropout": 0.0},
            {"d_model": 56, "max_sequence_length": 0, "dropout": 0.0},
            {"d_model": 56, "max_sequence_length": 60, "dropout": -0.1},
            {"d_model": 56, "max_sequence_length": 60, "dropout": 1.0},
        ]
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises((TypeError, ValueError)):
                    LearnablePositionalEncoding(**config)


if __name__ == "__main__":
    unittest.main()
