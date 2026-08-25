"""Interface, numerical, and gradient tests for strict-LCRT feature adaptation."""

from __future__ import annotations

import inspect
import math
import unittest

import torch
from torch import nn

from src.models.strict_lcrt_feature_adapter import StrictLCRTFeatureAdapter


FLOAT64_TOLERANCE = 5e-12
FLOAT32_FINITE_GRADIENT_FLOOR = 1e-10


class StrictLCRTFeatureAdapterTest(unittest.TestCase):
    """Verify real feature representations without changing strict LCRT math."""

    def setUp(self) -> None:
        """Use deterministic inputs throughout the adapter checks."""
        torch.manual_seed(20260825)

    def _adapter(
        self,
        representation: str = "input_real_imag",
        *,
        preserve_unselected: bool | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> StrictLCRTFeatureAdapter:
        """Build a non-Fourier adapter so both complex components are exercised."""
        module = StrictLCRTFeatureAdapter(
            selected_feature_indices=[1, 3],
            representation=representation,
            preserve_unselected=preserve_unselected,
            theta=math.pi / 3.0,
            scale=0.8,
        )
        return module.to(dtype=dtype)

    def test_representation_shapes_and_real_dtypes(self) -> None:
        """Default mode dimensions follow the documented selected-feature rules."""
        x = torch.randn(3, 60, 5, dtype=torch.float64)
        expected_features = {
            "real_imag": 4,
            "magnitude_phase": 4,
            "input_real_imag": 9,
        }
        for representation, feature_count in expected_features.items():
            with self.subTest(representation=representation):
                adapter = self._adapter(representation)
                output = adapter(x)
                self.assertEqual(tuple(output.shape), (3, 60, feature_count))
                self.assertEqual(output.dtype, torch.float64)
                self.assertFalse(torch.is_complex(output))
                self.assertEqual(
                    adapter.adapted_feature_count(5),
                    feature_count,
                )

    def test_real_imag_explicitly_retains_both_complex_components(self) -> None:
        """Cartesian adaptation must equal both components of the complex response."""
        x = torch.randn(2, 31, 5, dtype=torch.float64)
        adapter = self._adapter("real_imag")
        response = adapter.complex_response(x)
        cartesian = torch.view_as_real(response)
        expected = torch.cat((cartesian[..., 0], cartesian[..., 1]), dim=-1)
        self.assertTrue(torch.equal(adapter(x), expected))
        self.assertGreater(float(cartesian[..., 1].abs().max()), 1e-3)

        forward_source = inspect.getsource(StrictLCRTFeatureAdapter.forward)
        self.assertNotIn(".real", forward_source)

    def test_magnitude_phase_matches_principal_phase_without_unwrap(self) -> None:
        """Use nonnegative magnitude and principal torch.angle phase."""
        x = torch.randn(2, 31, 5, dtype=torch.float64)
        adapter = self._adapter("magnitude_phase", preserve_unselected=False)
        response = adapter.complex_response(x)
        output = adapter(x)
        magnitude, phase = output.chunk(2, dim=-1)
        self.assertTrue(torch.allclose(magnitude, response.abs(), atol=0.0, rtol=0.0))
        self.assertTrue(
            torch.allclose(phase, torch.angle(response), atol=0.0, rtol=0.0)
        )
        self.assertTrue(bool((magnitude >= 0).all().item()))
        self.assertTrue(bool((phase >= -torch.pi).all().item()))
        self.assertTrue(bool((phase <= torch.pi).all().item()))

    def test_zero_and_near_zero_phase_is_defined_as_zero(self) -> None:
        """Undefined phase is zeroed at amplitudes no larger than machine epsilon."""
        adapter = self._adapter("magnitude_phase")
        for value in (0.0, 1e-20):
            with self.subTest(value=value):
                x = torch.full((1, 32, 5), value, dtype=torch.float64)
                output = adapter(x)
                magnitude, phase = output.chunk(2, dim=-1)
                self.assertTrue(bool(torch.isfinite(output).all().item()))
                self.assertTrue(bool((magnitude >= 0).all().item()))
                self.assertEqual(float(phase.abs().max()), 0.0)

    def test_input_real_imag_preserves_all_original_features(self) -> None:
        """The recommended representation starts with the unchanged full input."""
        x = torch.randn(2, 33, 5, dtype=torch.float64)
        adapter = self._adapter("input_real_imag")
        output = adapter(x)
        self.assertTrue(torch.equal(output[..., :5], x))
        self.assertTrue(adapter.preserve_unselected)
        with self.assertRaisesRegex(ValueError, "inherently preserves"):
            self._adapter("input_real_imag", preserve_unselected=False)

    def test_other_modes_control_unselected_feature_preservation(self) -> None:
        """An explicit flag prepends unselected originals in their input order."""
        x = torch.randn(2, 32, 5, dtype=torch.float64)
        adapter = self._adapter("real_imag", preserve_unselected=True)
        output = adapter(x)
        self.assertEqual(tuple(output.shape), (2, 32, 7))
        self.assertEqual(adapter.adapted_feature_count(5), 7)
        self.assertTrue(torch.equal(output[..., :3], x[..., [0, 2, 4]]))

        without_originals = self._adapter(
            "real_imag",
            preserve_unselected=False,
        )(x)
        self.assertEqual(tuple(without_originals.shape), (2, 32, 4))

    def test_transform_operates_along_window_dimension(self) -> None:
        """Adapter transposition must match direct channel-first strict LCRT calls."""
        x = torch.zeros(2, 31, 5, dtype=torch.float64)
        x[0, 7, 1] = 1.0
        x[0, 19, 3] = -0.5
        x[1, 11, 1] = 2.0
        adapter = self._adapter("real_imag")
        selected = x[..., [1, 3]].transpose(1, 2).contiguous()
        expected = adapter.strict_lcrt(selected).transpose(1, 2).contiguous()
        self.assertTrue(
            torch.allclose(
                adapter.complex_response(x),
                expected,
                atol=FLOAT64_TOLERANCE,
                rtol=FLOAT64_TOLERANCE,
            )
        )

    def test_batch_samples_do_not_influence_each_other(self) -> None:
        """Processing windows together must equal processing each window alone."""
        x = torch.randn(4, 32, 5, dtype=torch.float64)
        adapter = self._adapter("input_real_imag")
        together = adapter(x)
        separately = torch.cat([adapter(sample[None]) for sample in x], dim=0)
        self.assertTrue(
            torch.allclose(
                together,
                separately,
                atol=FLOAT64_TOLERANCE,
                rtol=FLOAT64_TOLERANCE,
            )
        )

    def test_feature_channels_do_not_mix(self) -> None:
        """Perturbing one selected channel cannot change another channel response."""
        baseline = torch.randn(2, 31, 5, dtype=torch.float64)
        perturbed = baseline.clone()
        perturbed[..., 1] += torch.linspace(-3.0, 2.0, 31)
        adapter = self._adapter("real_imag")
        before = adapter.complex_response(baseline)
        after = adapter.complex_response(perturbed)
        self.assertTrue(torch.equal(before[..., 1], after[..., 1]))
        self.assertGreater(float((before[..., 0] - after[..., 0]).abs().max()), 1e-3)

    def test_input_and_lct_parameter_gradients_are_finite_and_nonzero(self) -> None:
        """Both Cartesian components retain gradients to inputs, theta, and scale."""
        x = torch.randn(3, 31, 5, dtype=torch.float64, requires_grad=True)
        adapter = self._adapter("input_real_imag")
        output = adapter(x)
        time_weights = torch.linspace(0.3, 1.7, 31, dtype=torch.float64)[None, :, None]
        feature_weights = torch.linspace(
            0.2,
            1.1,
            9,
            dtype=torch.float64,
        )[None, None, :]
        loss = (output.square() * time_weights * feature_weights).mean()
        loss.backward()

        gradients = (
            x.grad,
            adapter.strict_lcrt.lct.theta_unconstrained.grad,
            adapter.strict_lcrt.lct.log_scale.grad,
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(bool(torch.isfinite(gradient).all().item()))
            self.assertGreater(float(gradient.abs().max()), 1e-10)

    def test_supported_financial_signal_shapes_remain_finite(self) -> None:
        """Common window signals must not create NaN, Inf, or numerical explosion."""
        length = 33
        time = torch.arange(length, dtype=torch.float64)
        signals = {
            "zeros": torch.zeros(length, dtype=torch.float64),
            "constant": torch.ones(length, dtype=torch.float64),
            "impulse": torch.nn.functional.one_hot(
                torch.tensor(9), num_classes=length
            ).to(torch.float64),
            "trend": torch.linspace(-1.0, 1.0, length, dtype=torch.float64),
            "sine": torch.sin(2.0 * torch.pi * 4.0 * time / length),
            "random": torch.randn(length, dtype=torch.float64),
        }
        adapter = self._adapter("input_real_imag")
        for name, signal in signals.items():
            with self.subTest(signal=name):
                x = torch.zeros(1, length, 5, dtype=torch.float64)
                x[..., 1] = signal
                x[..., 3] = 0.3 * signal
                output = adapter(x)
                self.assertTrue(bool(torch.isfinite(output).all().item()))

    def test_different_feature_scales_do_not_explode(self) -> None:
        """The adapter remains finite and linear for differently scaled channels."""
        length = 60
        time = torch.arange(length, dtype=torch.float64)
        base = torch.sin(2.0 * torch.pi * 7.0 * time / length)
        x = torch.zeros(2, length, 5, dtype=torch.float64)
        x[..., 0] = 1e-6 * base
        x[..., 1] = base
        x[..., 2] = 1e3 * base
        x[..., 3] = 1e6 * base
        x[..., 4] = torch.linspace(-1e2, 1e2, length)
        adapter = self._adapter("real_imag")
        response = adapter.complex_response(x)
        self.assertTrue(bool(torch.isfinite(response).all().item()))
        input_norm = torch.linalg.vector_norm(x[..., [1, 3]], dim=1)
        output_norm = torch.linalg.vector_norm(response, dim=1)
        self.assertTrue(
            torch.allclose(
                output_norm,
                input_norm,
                atol=FLOAT64_TOLERANCE,
                rtol=FLOAT64_TOLERANCE,
            )
        )

    def test_float32_smoke_and_odd_even_window_batch_sizes(self) -> None:
        """Default precision supports odd/even windows and varied batches."""
        for batch_size in (1, 4):
            for window in (31, 32, 60):
                with self.subTest(batch_size=batch_size, window=window):
                    x = torch.randn(
                        batch_size,
                        window,
                        5,
                        dtype=torch.float32,
                    )
                    output = self._adapter(
                        "input_real_imag",
                        dtype=torch.float32,
                    )(x)
                    self.assertEqual(tuple(output.shape), (batch_size, window, 9))
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertTrue(bool(torch.isfinite(output).all().item()))

    def test_temporary_lstm_forward_backward_interface(self) -> None:
        """A temporary real LSTM receives finite gradients through the adapter."""
        adapter = self._adapter("input_real_imag", dtype=torch.float32)
        lstm = nn.LSTM(input_size=9, hidden_size=7, batch_first=True)
        x = torch.randn(3, 32, 5, dtype=torch.float32, requires_grad=True)
        adapted = adapter(x)
        recurrent, _ = lstm(adapted)
        loss = recurrent[:, -1].square().mean() + 0.03 * recurrent.mean()
        loss.backward()

        gradients = [
            x.grad,
            adapter.strict_lcrt.lct.theta_unconstrained.grad,
            adapter.strict_lcrt.lct.log_scale.grad,
            *(parameter.grad for parameter in lstm.parameters()),
        ]
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(bool(torch.isfinite(gradient).all().item()))
            self.assertGreater(
                float(gradient.abs().max()),
                FLOAT32_FINITE_GRADIENT_FLOOR,
            )

    def test_windows_share_no_statistics_or_state(self) -> None:
        """Adding a radically scaled window cannot change another window's output."""
        adapter = self._adapter("magnitude_phase")
        reference = torch.randn(1, 31, 5, dtype=torch.float64)
        unrelated = 1e6 * torch.randn(1, 31, 5, dtype=torch.float64)
        alone = adapter(reference)
        batched = adapter(torch.cat((reference, unrelated), dim=0))[:1]
        self.assertTrue(
            torch.allclose(
                alone,
                batched,
                atol=FLOAT64_TOLERANCE,
                rtol=FLOAT64_TOLERANCE,
            )
        )

    def test_invalid_interface_arguments_raise_clear_errors(self) -> None:
        """Feature selection, representation, shape, and dtype errors are explicit."""
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            StrictLCRTFeatureAdapter([])
        with self.assertRaisesRegex(ValueError, "negatives"):
            StrictLCRTFeatureAdapter([-1])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            StrictLCRTFeatureAdapter([1, 1])
        with self.assertRaisesRegex(ValueError, "representation"):
            StrictLCRTFeatureAdapter(
                [0],
                representation="real_only",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self._adapter()(torch.randn(2, 20, 3))
        with self.assertRaisesRegex(ValueError, "batch, window, features"):
            self._adapter()(torch.randn(20, 5))
        with self.assertRaisesRegex(TypeError, "float32 or torch.float64"):
            self._adapter()(torch.ones(2, 20, 5, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
