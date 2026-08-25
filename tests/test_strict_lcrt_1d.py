"""Numerical and structural tests for the paper-faithful strict LCRT baseline."""

from __future__ import annotations

import math
import unittest

import torch

from src.models.lct_riesz_1d import (
    fractional_riesz_multiplier as legacy_multiplier,
)
from src.models.lct_riesz_1d import lct_1d as legacy_lct_1d
from src.models.lct_riesz_1d import learnable_lct_matrix as legacy_lct_matrix
from src.models.strict_lcrt_1d import (
    StrictLCT1D,
    StrictLCRT1D,
    periodic_pv_hilbert_1d,
    strict_lcrt_multiplier,
)


FLOAT64_EXACT_TOLERANCE = 5e-12
FLOAT32_ROUNDTRIP_TOLERANCE = 2e-5


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return relative L2 error for a nonzero expected tensor."""
    return float(
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected)
    )


def _centered_fft(signal: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Return the centered orthonormal Fourier transform used by the strict module."""
    shifted = torch.fft.ifftshift(signal, dim=-1)
    if inverse:
        transformed = torch.fft.ifft(shifted, dim=-1, norm="ortho")
    else:
        transformed = torch.fft.fft(shifted, dim=-1, norm="ortho")
    return torch.fft.fftshift(transformed, dim=-1)


class StrictLCRT1DTest(unittest.TestCase):
    """Verify paper constraints, endpoint conventions, and discrete identities."""

    def test_a_equals_d_for_every_legal_parameter_set(self) -> None:
        """The two-parameter family must remain in the theorem's A=D class."""
        for theta, scale in (
            (math.pi / 6.0, 0.4),
            (math.pi / 3.0, 0.8),
            (2.0 * math.pi / 3.0, 1.7),
            (4.0 * math.pi / 3.0, 1.2),
            (5.0 * math.pi / 3.0, 2.0),
        ):
            module = StrictLCT1D(theta=theta, scale=scale).double()
            a, _, _, d = module.matrix()
            self.assertEqual(float((a - d).abs()), 0.0)

    def test_determinant_is_one_for_every_legal_parameter_set(self) -> None:
        """The strict matrix parameterization must analytically preserve determinant one."""
        for theta, scale in (
            (math.pi / 5.0, 0.6),
            (math.pi / 2.0, 1.0),
            (3.0 * math.pi / 4.0, 1.5),
            (5.0 * math.pi / 4.0, 0.9),
        ):
            a, b, c, d = StrictLCT1D(theta=theta, scale=scale).double().matrix()
            self.assertLess(float((a * d - b * c - 1.0).abs()), 2e-15)

    def test_scale_parameterization_is_strictly_positive(self) -> None:
        """Exponentiating log-scale must keep s positive without clipping."""
        module = StrictLCT1D(theta=math.pi / 3.0, scale=0.8).double()
        with torch.no_grad():
            module.log_scale.fill_(-50.0)
        self.assertGreater(float(module.scale()), 0.0)

    def test_fourier_parameters_produce_standard_matrix(self) -> None:
        """theta=pi/2 and s=1 must give [[0,1],[-1,0]]."""
        matrix = StrictLCT1D(theta=math.pi / 2.0, scale=1.0).double().matrix()
        expected = (0.0, 1.0, -1.0, 0.0)
        for actual, target in zip(matrix, expected):
            self.assertLess(abs(float(actual) - target), 2e-15)

    def test_negative_b_reverses_multiplier_sign(self) -> None:
        """The theorem's sign(omega/B) must flip every nonspecial bin for B<0."""
        positive = strict_lcrt_multiplier(
            64,
            1.0,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )
        negative = strict_lcrt_multiplier(
            64,
            -1.0,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )
        nonspecial = torch.ones(64, dtype=torch.bool)
        nonspecial[0] = False
        nonspecial[32] = False
        self.assertTrue(torch.equal(negative[nonspecial], -positive[nonspecial]))
        self.assertEqual(positive[33].item(), -1j)
        self.assertEqual(negative[33].item(), 1j)

    def test_dc_multiplier_is_zero(self) -> None:
        """The centered DC bin must be removed for either sign of B."""
        for b in (-1.2, 0.8):
            multiplier = strict_lcrt_multiplier(
                60,
                b,
                device=torch.device("cpu"),
                dtype=torch.complex128,
            )
            self.assertEqual(multiplier[30].item(), 0j)

    def test_even_nyquist_multiplier_is_zero(self) -> None:
        """The even-length Nyquist bin must be zero to preserve real symmetry."""
        for b in (-1.0, 1.0):
            multiplier = strict_lcrt_multiplier(
                64,
                b,
                device=torch.device("cpu"),
                dtype=torch.complex128,
            )
            self.assertEqual(multiplier[0].item(), 0j)

    def test_fourier_real_input_symmetry_and_output_imaginary_error(self) -> None:
        """The Fourier case must preserve real-signal symmetry and real Hilbert output."""
        torch.manual_seed(20260825)
        signal = torch.randn(3, 64, dtype=torch.float64)
        module = StrictLCRT1D(theta=math.pi / 2.0, scale=1.0).double()
        spectrum = module.lct(signal)
        canonical_phase = torch.polar(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(-math.pi / 4.0, dtype=torch.float64),
        ).to(torch.complex128)
        native = torch.fft.ifftshift(spectrum / canonical_phase, dim=-1)
        reverse = torch.remainder(-torch.arange(64), 64)
        self.assertTrue(
            torch.allclose(native, native.index_select(-1, reverse).conj(), atol=2e-12)
        )
        output = module(signal)
        self.assertLess(float(output.imag.abs().max()), 2e-12)

    def test_general_real_input_returns_theoretical_complex_response(self) -> None:
        """A non-Fourier LCRT must retain, rather than discard, its imaginary response."""
        indices = torch.arange(60, dtype=torch.float64) - 30
        signal = torch.exp(-((indices / 9.0) ** 2))
        output = StrictLCRT1D(theta=math.pi / 3.0, scale=0.8).double()(signal)
        self.assertTrue(torch.is_complex(output))
        self.assertGreater(float(output.imag.abs().max()), 1e-3)

    def test_lct_roundtrip_is_accurate_in_float64_and_float32(self) -> None:
        """Use dtype-specific tolerances based on FFT rounding and precision."""
        torch.manual_seed(20260825)
        for theta, scale in (
            (math.pi / 3.0, 0.8),
            (2.0 * math.pi / 3.0, 1.2),
            (4.0 * math.pi / 3.0, 1.1),
        ):
            for length in (31, 32, 60, 64):
                module64 = StrictLCT1D(theta=theta, scale=scale).double()
                signal64 = torch.randn(2, length, dtype=torch.float64)
                recovered64 = module64(module64(signal64), inverse=True)
                self.assertLess(
                    _relative_l2(recovered64, signal64),
                    FLOAT64_EXACT_TOLERANCE,
                )

                module32 = StrictLCT1D(theta=theta, scale=scale).float()
                signal32 = signal64.float()
                recovered32 = module32(module32(signal32), inverse=True)
                self.assertLess(
                    _relative_l2(recovered32, signal32),
                    FLOAT32_ROUNDTRIP_TOLERANCE,
                )

    def test_fourier_case_matches_fft_and_classical_hilbert(self) -> None:
        """Verify the exact fixed phase/order and the classical Hilbert endpoint."""
        torch.manual_seed(20260825)
        signal = torch.randn(2, 60, dtype=torch.float64)
        module = StrictLCRT1D(theta=math.pi / 2.0, scale=1.0).double()
        phase = torch.polar(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(-math.pi / 4.0, dtype=torch.float64),
        ).to(torch.complex128)
        expected_lct = phase * _centered_fft(signal.to(torch.complex128))
        self.assertLess(
            _relative_l2(module.lct(signal), expected_lct),
            FLOAT64_EXACT_TOLERANCE,
        )

        native_frequencies = torch.fft.fftfreq(60, dtype=torch.float64)
        hilbert_multiplier = (-1j * torch.sign(native_frequencies)).to(
            torch.complex128
        )
        hilbert_multiplier[0] = 0
        hilbert_multiplier[30] = 0
        expected_hilbert = torch.fft.ifft(
            torch.fft.fft(signal) * hilbert_multiplier
        )
        self.assertLess(
            _relative_l2(module(signal), expected_hilbert),
            FLOAT64_EXACT_TOLERANCE,
        )

    def test_spatial_reference_matches_multiplier_for_a_equal_d(self) -> None:
        """Compare independent periodic-PV and LCT-multiplier discretizations."""
        torch.manual_seed(20260825)
        for theta, scale in (
            (math.pi / 3.0, 0.8),
            (2.0 * math.pi / 3.0, 1.2),
            (4.0 * math.pi / 3.0, 1.1),
        ):
            module = StrictLCRT1D(theta=theta, scale=scale).double()
            for length in (31, 32, 60):
                signal = torch.randn(2, length, dtype=torch.float64)
                multiplier_output = module(signal)
                spatial_output = module.spatial_reference(signal)
                self.assertLess(
                    _relative_l2(multiplier_output, spatial_output),
                    FLOAT64_EXACT_TOLERANCE,
                )

    def test_periodic_principal_value_excludes_diagonal(self) -> None:
        """The PV reference must annihilate constants after excluding y=x."""
        constant = torch.ones(2, 32, dtype=torch.float64)
        output = periodic_pv_hilbert_1d(constant)
        self.assertLess(float(output.abs().max()), 2e-15)

    def _parameter_gradients(self) -> tuple[float, float]:
        """Return internal theta and log-scale gradients for one asymmetric loss."""
        torch.manual_seed(20260825)
        module = StrictLCRT1D(theta=math.pi / 3.0, scale=0.8).double()
        signal = torch.randn(3, 60, dtype=torch.float64)
        weights = torch.linspace(0.2, 1.4, 60, dtype=torch.float64)
        output = module(signal)
        loss = (output.abs().square() * weights).mean() + 0.07 * output.real.mean()
        loss.backward()
        theta_gradient = module.lct.theta_unconstrained.grad
        scale_gradient = module.lct.log_scale.grad
        self.assertIsNotNone(theta_gradient)
        self.assertIsNotNone(scale_gradient)
        assert theta_gradient is not None
        assert scale_gradient is not None
        return float(theta_gradient), float(scale_gradient)

    def test_theta_gradient_is_finite_and_nonzero(self) -> None:
        """The constrained physical angle must retain a useful training gradient."""
        theta_gradient, _ = self._parameter_gradients()
        self.assertTrue(math.isfinite(theta_gradient))
        self.assertGreater(abs(theta_gradient), 1e-10)

    def test_scale_gradient_is_finite_and_nonzero(self) -> None:
        """The positive log-scale parameter must retain a useful training gradient."""
        _, scale_gradient = self._parameter_gradients()
        self.assertTrue(math.isfinite(scale_gradient))
        self.assertGreater(abs(scale_gradient), 1e-10)

    def test_module_has_no_q_gamma_gate_or_fusion_parameters(self) -> None:
        """The strict baseline must contain only theta and scale trainable parameters."""
        module = StrictLCRT1D()
        names = {name for name, _ in module.named_parameters()}
        self.assertEqual(names, {"lct.theta_unconstrained", "lct.log_scale"})
        self.assertEqual(set(module.export_parameters()), {"theta", "s", "A", "B", "C", "D"})

    def test_near_zero_b_is_rejected_without_clamping(self) -> None:
        """Illegal or near-singular B values must raise explicit exceptions."""
        for theta in (0.0, math.pi, 2.0 * math.pi):
            with self.assertRaisesRegex(ValueError, "theta"):
                StrictLCT1D(theta=theta, scale=1.0)
        with self.assertRaisesRegex(ValueError, r"\|B\|"):
            StrictLCT1D(
                theta=math.pi / 2.0,
                scale=1e-12,
                b_epsilon=1e-8,
            )

    def test_legacy_module_fourier_and_nyquist_behavior_is_unchanged(self) -> None:
        """Guard the historical module's established Fourier and Nyquist conventions."""
        torch.manual_seed(20260825)
        signal = torch.randn(60, dtype=torch.float64)
        matrix = legacy_lct_matrix(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
        )
        coefficient = torch.polar(
            torch.tensor(1.0 / math.sqrt(60), dtype=torch.float64),
            torch.tensor(-math.pi / 4.0, dtype=torch.float64),
        ).to(torch.complex128)
        self.assertTrue(
            torch.allclose(
                legacy_lct_1d(signal, matrix),
                coefficient * torch.fft.fft(signal),
                atol=2e-12,
                rtol=2e-12,
            )
        )
        multiplier = legacy_multiplier(
            64,
            1.0,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )
        self.assertAlmostEqual(float(multiplier[32].imag), 1.0, places=14)


if __name__ == "__main__":
    unittest.main()
