"""Minimal tensor tests for the one-dimensional learnable LCT-Riesz module."""

from __future__ import annotations

import math
import unittest

import torch

from src.models.lct_riesz_1d import (
    LearnableLCT1D,
    LearnableLCTRiesz1D,
    fractional_riesz_multiplier,
    inverse_lct_matrix,
    lct_1d,
    learnable_lct_matrix,
)


class LearnableLCTRiesz1DTest(unittest.TestCase):
    """Verify shape preservation, gradients, and parameter export."""

    def test_forward_backward_and_parameter_export(self) -> None:
        """Run a complete differentiable pass on an OHLCV-shaped tensor."""
        torch.manual_seed(42)
        x = torch.randn(4, 5, 60)
        module = LearnableLCTRiesz1D(channels=5, gate_init=1.0)

        output = module(x)
        self.assertEqual(output.shape, (4, 5, 60))

        loss = output.mean()
        loss.backward()

        learnable_parameters = {
            "alpha_t": module.lct.alpha_t,
            "log_m_t": module.lct.log_m_t,
            "q_t": module.lct.q_t,
        }
        for name, parameter in learnable_parameters.items():
            self.assertIsNotNone(parameter.grad, f"{name} has no gradient.")
            self.assertTrue(
                torch.isfinite(parameter.grad).all(),
                f"{name} has a non-finite gradient.",
            )

        exported = module.export_parameters()
        expected_keys = {"alpha", "m", "q", "A", "B", "C", "D", "gamma"}
        self.assertEqual(set(exported), expected_keys)
        self.assertTrue(all(isinstance(value, float) for value in exported.values()))
        self.assertAlmostEqual(
            exported["A"] * exported["D"] - exported["B"] * exported["C"],
            1.0,
            places=5,
        )

    def test_fourier_special_case_matches_unitary_fft_with_fixed_phase(self) -> None:
        """Verify the exact discrete normalization and phase for even lengths."""
        torch.manual_seed(20260825)
        module = LearnableLCT1D(alpha=1.0, m=1.0, q=0.0).double()
        for length in (32, 60, 64, 128):
            signal = torch.randn(3, length, dtype=torch.float64)
            actual = module(signal)
            coefficient = torch.polar(
                torch.tensor(1.0 / math.sqrt(length), dtype=torch.float64),
                torch.tensor(-math.pi / 4.0, dtype=torch.float64),
            ).to(torch.complex128)
            expected = coefficient * torch.fft.fft(signal, dim=-1)
            self.assertTrue(
                torch.allclose(actual, expected, rtol=2e-12, atol=2e-12),
                f"Fourier convention mismatch at length {length}.",
            )

    def test_forward_inverse_roundtrip_is_double_precision_accurate(self) -> None:
        """Check regular-branch reconstruction including non-power-of-two length 60."""
        torch.manual_seed(20260825)
        parameter_sets = (
            (1.0, 1.0, 0.0),
            (0.25, 1.0, 0.0),
            (0.5, 1.0, 0.0),
            (0.75, 1.0, 0.0),
            (0.5, 0.8, -0.5),
            (0.5, 1.2, 0.5),
        )
        for length in (32, 60, 64, 128):
            signal = torch.randn(2, length, dtype=torch.float64)
            for alpha, m, q in parameter_sets:
                matrix = learnable_lct_matrix(
                    torch.tensor(alpha, dtype=torch.float64),
                    torch.tensor(math.log(m), dtype=torch.float64),
                    torch.tensor(q, dtype=torch.float64),
                )
                reconstructed = lct_1d(
                    lct_1d(signal, matrix),
                    inverse_lct_matrix(matrix),
                )
                relative_l2 = torch.linalg.vector_norm(
                    reconstructed - signal
                ) / torch.linalg.vector_norm(signal)
                self.assertLess(
                    float(relative_l2),
                    1e-11,
                    f"Roundtrip mismatch for N={length}, alpha={alpha}, m={m}, q={q}.",
                )

    def test_project_hilbert_extension_matches_formula_away_from_dc(self) -> None:
        """Check the project extension formula without attributing it to the paper."""
        for length in (32, 60, 64, 128):
            frequencies = torch.fft.fftfreq(length, dtype=torch.float64)
            signs = torch.sign(frequencies)
            non_dc = frequencies != 0
            for gamma in (0.0, 0.25, 0.5, 0.75, 1.0):
                actual = fractional_riesz_multiplier(
                    length,
                    gamma,
                    device=torch.device("cpu"),
                    dtype=torch.complex128,
                )
                phase = torch.tensor(gamma * math.pi / 2.0, dtype=torch.float64)
                expected = (
                    torch.cos(phase) - 1j * signs * torch.sin(phase)
                ).to(torch.complex128)
                self.assertTrue(
                    torch.allclose(
                        actual[non_dc],
                        expected[non_dc],
                        rtol=0.0,
                        atol=1e-14,
                    )
                )
                self.assertEqual(actual[0].item(), 0j)

    def test_gamma_zero_is_mean_removal_not_strict_identity(self) -> None:
        """Record that the current gamma-zero endpoint deletes the DC component."""
        signal = torch.tensor([2.0, 3.0, 5.0, 7.0], dtype=torch.float64)
        multiplier = fractional_riesz_multiplier(
            signal.numel(),
            0.0,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )
        output = torch.fft.ifft(torch.fft.fft(signal) * multiplier).real
        expected_zero_mean = signal - signal.mean()
        self.assertTrue(torch.allclose(output, expected_zero_mean, atol=1e-14, rtol=0.0))
        self.assertFalse(torch.allclose(output, signal, atol=1e-14, rtol=0.0))

    def test_gamma_one_dc_zero_matches_discrete_hilbert_convention(self) -> None:
        """Record DC zeroing as valid for the classical gamma-one Hilbert endpoint."""
        multiplier = fractional_riesz_multiplier(
            60,
            1.0,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )
        self.assertEqual(multiplier[0].item(), 0j)

        constant = torch.full((60,), 3.5, dtype=torch.float64)
        output = torch.fft.ifft(torch.fft.fft(constant) * multiplier)
        self.assertTrue(torch.allclose(output, torch.zeros_like(output), atol=1e-14))

    def test_even_nyquist_uses_negative_fftfreq_sign(self) -> None:
        """Record the current +i gamma-one Nyquist convention for even lengths."""
        multiplier = fractional_riesz_multiplier(
            64,
            1.0,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )
        self.assertAlmostEqual(float(multiplier[32].real), 0.0, places=14)
        self.assertAlmostEqual(float(multiplier[32].imag), 1.0, places=14)

    def test_open_gate_exposes_q_gradient_cancellation(self) -> None:
        """Verify active gradients while documenting the numerically inactive q path."""
        torch.manual_seed(20260825)
        module = LearnableLCTRiesz1D(
            channels=3,
            alpha=0.73,
            m=1.08,
            q=0.17,
            gamma=0.6,
            learnable_gamma=True,
            gate_init=1.0,
        ).double()
        signal = torch.randn(4, 3, 60, dtype=torch.float64)
        weights = torch.linspace(0.25, 1.25, 60, dtype=torch.float64)
        output = module(signal)
        loss = (output.square() * weights.view(1, 1, -1)).mean()
        loss.backward()
        parameters = {
            "alpha_t": module.lct.alpha_t,
            "log_m_t": module.lct.log_m_t,
            "q_t": module.lct.q_t,
            "gamma": module.gamma,
            "gate": module.gate,
        }
        for name, parameter in parameters.items():
            self.assertIsNotNone(parameter.grad, f"{name} gradient is None.")
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for name in ("alpha_t", "log_m_t", "gamma", "gate"):
            self.assertGreater(
                float(parameters[name].grad.abs().max()),
                1e-12,
                f"{name} gradient is numerically zero with an open gate.",
            )
        self.assertLess(
            float(parameters["q_t"].grad.abs().max()),
            1e-12,
            "q_t unexpectedly affects the composed LCT-Riesz-inverse-LCT path.",
        )

    def test_q_changes_do_not_affect_complete_conjugated_output(self) -> None:
        """Confirm q is algebraically redundant in the current complete block."""
        torch.manual_seed(20260825)
        module = LearnableLCTRiesz1D(
            channels=2,
            alpha=0.73,
            m=1.08,
            q=0.0,
            gamma=0.6,
            learnable_gamma=False,
            gate_init=1.0,
        ).double()
        signal = torch.randn(3, 2, 60, dtype=torch.float64)
        outputs = []
        with torch.no_grad():
            for q_value in (-0.5, 0.0, 0.5):
                module.lct.q_t.fill_(q_value)
                outputs.append(module(signal))
        for output in outputs[1:]:
            self.assertTrue(
                torch.allclose(output, outputs[0], rtol=1e-12, atol=1e-12),
                "q changed the current LCT-multiplier-inverse-LCT output.",
            )

    def test_general_parameterization_does_not_enforce_a_equal_d(self) -> None:
        """Record why a general learned matrix need not meet the paper theorem."""
        matrix = learnable_lct_matrix(
            torch.tensor(0.73, dtype=torch.float64),
            torch.tensor(math.log(1.08), dtype=torch.float64),
            torch.tensor(0.17, dtype=torch.float64),
        )
        self.assertGreater(float((matrix[0] - matrix[3]).abs()), 1e-6)

    def test_fourier_parameters_satisfy_a_equal_d(self) -> None:
        """Verify the Fourier initialization lies in the paper theorem's A=D class."""
        matrix = learnable_lct_matrix(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
        )
        self.assertLess(float((matrix[0] - matrix[3]).abs()), 1e-14)


if __name__ == "__main__":
    unittest.main()
