"""Validate the paper-faithful one-dimensional strict LCRT baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.models.strict_lcrt_1d import (
    StrictLCT1D,
    StrictLCRT1D,
    strict_lcrt_multiplier,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "numerical_validation" / "strict_lcrt"
SEED = 20260825
LENGTHS = (31, 32, 60, 64, 128)
PARAMETER_SETS = (
    ("fourier", math.pi / 2.0, 1.0),
    ("positive_b_fractional", math.pi / 3.0, 0.8),
    ("positive_b_scaled", 2.0 * math.pi / 3.0, 1.2),
    ("negative_b_fractional", 4.0 * math.pi / 3.0, 1.1),
    ("negative_b_scaled", 5.0 * math.pi / 3.0, 0.9),
)
FLOAT64_EXACT_TOLERANCE = 1e-11
MATRIX_TOLERANCE = 1e-12
GRADIENT_MINIMUM = 1e-10


def _as_float(value: torch.Tensor | float) -> float:
    """Return one detached scalar as a Python float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return relative L2 error, including a safe zero-reference branch."""
    numerator = torch.linalg.vector_norm(actual - expected)
    denominator = torch.linalg.vector_norm(expected)
    if _as_float(denominator) == 0.0:
        return _as_float(numerator)
    return _as_float(numerator / denominator)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Return maximum absolute and relative L2 errors."""
    return {
        "max_abs_error": _as_float((actual - expected).abs().max()),
        "relative_l2_error": _relative_l2(actual, expected),
    }


def _signals(length: int) -> dict[str, torch.Tensor]:
    """Build deterministic smooth, oscillatory, and random real signals."""
    indices = torch.arange(length, dtype=torch.float64) - length // 2
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + length)
    impulse = torch.zeros(length, dtype=torch.float64)
    impulse[length // 2] = 1.0
    return {
        "centered_impulse": impulse,
        "single_sine": torch.sin(2.0 * torch.pi * 3.0 * indices / length),
        "dual_frequency": (
            0.8 * torch.cos(2.0 * torch.pi * 3.0 * indices / length)
            + 0.35 * torch.sin(2.0 * torch.pi * 7.0 * indices / length)
        ),
        "random_real": torch.randn(length, generator=generator, dtype=torch.float64),
        "smooth_chirp": torch.exp(-((indices / (0.18 * length)) ** 2))
        * torch.cos(
            2.0
            * torch.pi
            * (0.04 * indices + 0.18 * indices.square() / (2.0 * length))
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a homogeneous UTF-8 CSV validation table."""
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    """Convert paths, tensors, mappings, and nonfinite values for JSON."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a readable UTF-8 JSON validation artifact."""
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _centered_fft(signal: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Apply the validation reference centered orthonormal FFT or IFFT."""
    shifted = torch.fft.ifftshift(signal, dim=-1)
    if inverse:
        transformed = torch.fft.ifft(shifted, dim=-1, norm="ortho")
    else:
        transformed = torch.fft.fft(shifted, dim=-1, norm="ortho")
    return torch.fft.fftshift(transformed, dim=-1)


def validate_matrices() -> list[dict[str, Any]]:
    """Check A=D, determinant one, positive scale, and nonzero B."""
    rows: list[dict[str, Any]] = []
    for name, theta, scale in PARAMETER_SETS:
        module = StrictLCT1D(theta=theta, scale=scale).double()
        values = module.export_parameters()
        determinant = values["A"] * values["D"] - values["B"] * values["C"]
        rows.append(
            {
                "parameter_set": name,
                "theta": values["theta"],
                "s": values["s"],
                "A": values["A"],
                "B": values["B"],
                "C": values["C"],
                "D": values["D"],
                "abs_A_minus_D": abs(values["A"] - values["D"]),
                "abs_determinant_minus_one": abs(determinant - 1.0),
                "scale_positive": values["s"] > 0.0,
                "B_nonzero": abs(values["B"]) >= module.b_epsilon,
            }
        )
    return rows


def validate_fourier_case() -> list[dict[str, Any]]:
    """Compare the Fourier matrix with centered FFT and classical Hilbert output."""
    module = StrictLCRT1D(theta=math.pi / 2.0, scale=1.0).double()
    phase = torch.polar(
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(-math.pi / 4.0, dtype=torch.float64),
    ).to(torch.complex128)
    rows: list[dict[str, Any]] = []
    for length in LENGTHS:
        native_frequencies = torch.fft.fftfreq(length, dtype=torch.float64)
        hilbert_multiplier = (-1j * torch.sign(native_frequencies)).to(
            torch.complex128
        )
        hilbert_multiplier[0] = 0
        if length % 2 == 0:
            hilbert_multiplier[length // 2] = 0
        for signal_name, signal in _signals(length).items():
            actual_lct = module.lct(signal)
            expected_lct = phase * _centered_fft(signal.to(torch.complex128))
            lct_metrics = _metrics(actual_lct, expected_lct)
            actual_hilbert = module(signal)
            expected_hilbert = torch.fft.ifft(
                torch.fft.fft(signal) * hilbert_multiplier
            )
            hilbert_metrics = _metrics(actual_hilbert, expected_hilbert)
            rows.append(
                {
                    "length": length,
                    "signal": signal_name,
                    "lct_fft_max_abs_error": lct_metrics["max_abs_error"],
                    "lct_fft_relative_l2_error": lct_metrics["relative_l2_error"],
                    "lcrt_hilbert_max_abs_error": hilbert_metrics["max_abs_error"],
                    "lcrt_hilbert_relative_l2_error": hilbert_metrics[
                        "relative_l2_error"
                    ],
                    "max_output_imaginary_abs": _as_float(
                        actual_hilbert.imag.abs().max()
                    ),
                }
            )
    return rows


def validate_roundtrip() -> list[dict[str, Any]]:
    """Measure strict LCT forward/inverse reconstruction over signs and lengths."""
    rows: list[dict[str, Any]] = []
    for name, theta, scale in PARAMETER_SETS:
        module = StrictLCT1D(theta=theta, scale=scale).double()
        b_value = module.export_parameters()["B"]
        for length in LENGTHS:
            for signal_name, signal in _signals(length).items():
                reconstructed = module(module(signal), inverse=True)
                metrics = _metrics(reconstructed, signal.to(torch.complex128))
                rows.append(
                    {
                        "parameter_set": name,
                        "theta": theta,
                        "s": scale,
                        "B": b_value,
                        "length": length,
                        "signal": signal_name,
                        **metrics,
                    }
                )
    return rows


def validate_spatial_multiplier_alignment() -> list[dict[str, Any]]:
    """Compare periodic spatial PV and Equation (2.1) without fitted corrections."""
    rows: list[dict[str, Any]] = []
    small_lengths = (31, 32, 60, 64)
    for name, theta, scale in PARAMETER_SETS:
        module = StrictLCRT1D(theta=theta, scale=scale).double()
        for length in small_lengths:
            for signal_name, signal in _signals(length).items():
                multiplier_output = module(signal)
                spatial_output = module.spatial_reference(signal)
                metrics = _metrics(multiplier_output, spatial_output)
                rows.append(
                    {
                        "parameter_set": name,
                        "theta": theta,
                        "s": scale,
                        "B": module.export_parameters()["B"],
                        "length": length,
                        "signal": signal_name,
                        "boundary": "periodic",
                        "sample_spacing": math.sqrt(2.0 * math.pi / length),
                        "diagonal_pv_term_excluded": True,
                        "fitted_scale_or_phase": False,
                        **metrics,
                    }
                )
    return rows


def validate_multiplier_endpoints() -> list[dict[str, Any]]:
    """Record formula, DC, Nyquist, and B-sign behavior bin by bin."""
    rows: list[dict[str, Any]] = []
    for length in LENGTHS:
        frequencies = torch.fft.fftshift(torch.fft.fftfreq(length, dtype=torch.float64))
        for b in (-1.2, 0.8):
            actual = strict_lcrt_multiplier(
                length,
                b,
                device=torch.device("cpu"),
                dtype=torch.complex128,
            )
            expected = (-1j * torch.sign(frequencies / b)).to(torch.complex128)
            expected[frequencies == 0] = 0
            if length % 2 == 0:
                expected[0] = 0
            for index in range(length):
                rows.append(
                    {
                        "length": length,
                        "B": b,
                        "bin": index,
                        "frequency": _as_float(frequencies[index]),
                        "is_dc": bool(frequencies[index] == 0),
                        "is_even_nyquist": length % 2 == 0 and index == 0,
                        "actual_real": _as_float(actual[index].real),
                        "actual_imag": _as_float(actual[index].imag),
                        "expected_real": _as_float(expected[index].real),
                        "expected_imag": _as_float(expected[index].imag),
                        "complex_abs_error": _as_float(
                            (actual[index] - expected[index]).abs()
                        ),
                    }
                )
    return rows


def validate_gradients() -> dict[str, Any]:
    """Backpropagate once without an optimizer step and report physical gradients."""
    torch.manual_seed(SEED)
    module = StrictLCRT1D(theta=math.pi / 3.0, scale=0.8).double()
    signal = torch.randn(4, 60, dtype=torch.float64)
    weights = torch.linspace(0.2, 1.4, 60, dtype=torch.float64)
    output = module(signal)
    loss = (output.abs().square() * weights).mean() + 0.07 * output.real.mean()
    loss.backward()
    theta_raw_gradient = module.lct.theta_unconstrained.grad
    log_scale_gradient = module.lct.log_scale.grad
    if theta_raw_gradient is None or log_scale_gradient is None:
        raise RuntimeError("Strict LCRT parameters did not receive gradients.")
    sigmoid_value = torch.sigmoid(module.lct.theta_unconstrained.detach())
    theta_interval = math.pi - 2.0 * module.lct.theta_margin
    dtheta_draw = theta_interval * sigmoid_value * (1.0 - sigmoid_value)
    theta_gradient = theta_raw_gradient.detach() / dtheta_draw
    scale_gradient = log_scale_gradient.detach() / module.lct.scale().detach()
    return {
        "loss": _as_float(loss),
        "theta": _as_float(module.lct.theta()),
        "s": _as_float(module.lct.scale()),
        "theta_unconstrained_gradient": _as_float(theta_raw_gradient),
        "theta_physical_gradient": _as_float(theta_gradient),
        "log_scale_gradient": _as_float(log_scale_gradient),
        "scale_physical_gradient": _as_float(scale_gradient),
        "theta_gradient_finite": bool(torch.isfinite(theta_raw_gradient).item()),
        "scale_gradient_finite": bool(torch.isfinite(log_scale_gradient).item()),
        "optimizer_step_performed": False,
    }


def _git_value(*arguments: str) -> str:
    """Return one read-only Git provenance value."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_summary(
    *,
    output_dir: Path,
    matrix_rows: Sequence[Mapping[str, Any]],
    fourier_rows: Sequence[Mapping[str, Any]],
    roundtrip_rows: Sequence[Mapping[str, Any]],
    spatial_rows: Sequence[Mapping[str, Any]],
    multiplier_rows: Sequence[Mapping[str, Any]],
    gradients: Mapping[str, Any],
) -> dict[str, Any]:
    """Build per-component PASS/FAIL conclusions from fixed tolerances."""
    max_a_minus_d = max(float(row["abs_A_minus_D"]) for row in matrix_rows)
    max_determinant_error = max(
        float(row["abs_determinant_minus_one"]) for row in matrix_rows
    )
    max_fourier_error = max(
        float(row["lct_fft_relative_l2_error"]) for row in fourier_rows
    )
    max_hilbert_error = max(
        float(row["lcrt_hilbert_relative_l2_error"]) for row in fourier_rows
    )
    max_roundtrip_error = max(
        float(row["relative_l2_error"]) for row in roundtrip_rows
    )
    max_spatial_error = max(
        float(row["relative_l2_error"]) for row in spatial_rows
    )
    max_multiplier_error = max(
        float(row["complex_abs_error"]) for row in multiplier_rows
    )
    max_fourier_imaginary = max(
        float(row["max_output_imaginary_abs"]) for row in fourier_rows
    )
    general_imaginary = []
    for _, theta, scale in PARAMETER_SETS:
        if abs(theta - math.pi / 2.0) < 1e-12:
            continue
        module = StrictLCRT1D(theta=theta, scale=scale).double()
        output = module(_signals(60)["random_real"])
        general_imaginary.append(_as_float(output.imag.abs().max()))

    checks = {
        "discrete_lct_core": (
            max_fourier_error <= FLOAT64_EXACT_TOLERANCE
            and max_roundtrip_error <= FLOAT64_EXACT_TOLERANCE
        ),
        "paper_matrix_conditions": (
            max_a_minus_d <= MATRIX_TOLERANCE
            and max_determinant_error <= MATRIX_TOLERANCE
            and all(bool(row["scale_positive"]) for row in matrix_rows)
            and all(bool(row["B_nonzero"]) for row in matrix_rows)
        ),
        "paper_multiplier": (
            max_multiplier_error <= FLOAT64_EXACT_TOLERANCE
            and max_hilbert_error <= FLOAT64_EXACT_TOLERANCE
        ),
        "spatial_multiplier_discrete_alignment": (
            max_spatial_error <= FLOAT64_EXACT_TOLERANCE
        ),
        "fourier_real_output": max_fourier_imaginary <= FLOAT64_EXACT_TOLERANCE,
        "theta_gradient": (
            bool(gradients["theta_gradient_finite"])
            and abs(float(gradients["theta_unconstrained_gradient"]))
            > GRADIENT_MINIMUM
        ),
        "scale_gradient": (
            bool(gradients["scale_gradient_finite"])
            and abs(float(gradients["log_scale_gradient"])) > GRADIENT_MINIMUM
        ),
    }
    ready = all(checks.values())
    return {
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "output_dir": str(output_dir.relative_to(ROOT)),
            "git_branch": _git_value("branch", "--show-current"),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "float64/complex128",
            "seed": SEED,
            "training_performed": False,
            "optimizer_step_performed": False,
        },
        "definition": {
            "matrix": "A=D=cos(theta), B=s*sin(theta), C=-sin(theta)/s",
            "scale_parameterization": "s=exp(log_scale)>0",
            "multiplier": "-i*sign(omega/B)",
            "gamma": "fixed to the paper Riesz/Hilbert endpoint; no gamma parameter",
            "dc": "zero",
            "even_nyquist": "zero to preserve real-input conjugate symmetry",
            "output": (
                "complex in general; Fourier-matrix real-input output is real up to "
                "rounding, and no .real projection is applied"
            ),
        },
        "thresholds": {
            "float64_exact_identity": FLOAT64_EXACT_TOLERANCE,
            "matrix": MATRIX_TOLERANCE,
            "gradient_minimum": GRADIENT_MINIMUM,
            "rationale": (
                "FFT, inverse, multiplier, and periodic spectral-PV identities use "
                "float64/complex128 and should agree within accumulated rounding. "
                "No fitted scale, phase, or frequency-bin correction is used."
            ),
        },
        "maximum_errors": {
            "abs_A_minus_D": max_a_minus_d,
            "abs_determinant_minus_one": max_determinant_error,
            "fourier_lct_relative_l2": max_fourier_error,
            "fourier_hilbert_relative_l2": max_hilbert_error,
            "roundtrip_relative_l2": max_roundtrip_error,
            "spatial_vs_multiplier_relative_l2": max_spatial_error,
            "multiplier_complex_abs": max_multiplier_error,
            "fourier_output_imaginary_abs": max_fourier_imaginary,
            "general_nonfourier_output_imaginary_abs_range": [
                min(general_imaginary),
                max(general_imaginary),
            ],
        },
        "gradients": gradients,
        "endpoint_values": {
            "positive_B_DC": next(
                [row["actual_real"], row["actual_imag"]]
                for row in multiplier_rows
                if float(row["B"]) > 0 and bool(row["is_dc"])
            ),
            "negative_B_DC": next(
                [row["actual_real"], row["actual_imag"]]
                for row in multiplier_rows
                if float(row["B"]) < 0 and bool(row["is_dc"])
            ),
            "positive_B_even_nyquist": next(
                [row["actual_real"], row["actual_imag"]]
                for row in multiplier_rows
                if float(row["B"]) > 0 and bool(row["is_even_nyquist"])
            ),
            "negative_B_even_nyquist": next(
                [row["actual_real"], row["actual_imag"]]
                for row in multiplier_rows
                if float(row["B"]) < 0 and bool(row["is_even_nyquist"])
            ),
        },
        "checks": {
            name: "PASS" if passed else "FAIL" for name, passed in checks.items()
        },
        "overall_conclusion": {
            "discrete_lct_core": (
                "PASS" if checks["discrete_lct_core"] else "FAIL"
            ),
            "paper_matrix_conditions": (
                "PASS" if checks["paper_matrix_conditions"] else "FAIL"
            ),
            "paper_multiplier": "PASS" if checks["paper_multiplier"] else "FAIL",
            "spatial_multiplier_discrete_consistency": (
                "PASS"
                if checks["spatial_multiplier_discrete_alignment"]
                else "FAIL"
            ),
            "ready_for_prediction_model_integration": ready,
            "integration_status": (
                "Numerically ready for a future controlled integration, but this "
                "task intentionally does not connect the module to any forecaster."
                if ready
                else "Not ready; resolve failed checks before model integration."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse an optional explicit output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> Path:
    """Run all strict LCRT validations and write a fresh timestamped directory."""
    args = parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    matrix_rows = validate_matrices()
    fourier_rows = validate_fourier_case()
    roundtrip_rows = validate_roundtrip()
    spatial_rows = validate_spatial_multiplier_alignment()
    multiplier_rows = validate_multiplier_endpoints()
    gradients = validate_gradients()
    summary = build_summary(
        output_dir=output_dir,
        matrix_rows=matrix_rows,
        fourier_rows=fourier_rows,
        roundtrip_rows=roundtrip_rows,
        spatial_rows=spatial_rows,
        multiplier_rows=multiplier_rows,
        gradients=gradients,
    )

    _write_csv(output_dir / "matrix_validation.csv", matrix_rows)
    _write_csv(output_dir / "fourier_alignment.csv", fourier_rows)
    _write_csv(output_dir / "roundtrip.csv", roundtrip_rows)
    _write_csv(output_dir / "spatial_multiplier_alignment.csv", spatial_rows)
    _write_csv(output_dir / "multiplier_endpoint_checks.csv", multiplier_rows)
    _write_json(output_dir / "gradient_diagnostics.json", gradients)
    _write_json(output_dir / "validation_summary.json", summary)
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
