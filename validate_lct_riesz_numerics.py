"""Validate the existing one-dimensional LCT and project Hilbert extension.

This diagnostic intentionally does not train forecasting models or alter the
production transform implementation.  It evaluates the existing functions in
``src.models.lct_riesz_1d`` using float64/complex128 arithmetic and writes a
timestamped, ignored artifact directory for audit and thesis use.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from src.models.lct_riesz_1d import (
    LearnableLCTRiesz1D,
    fractional_riesz_multiplier,
    inverse_lct_matrix,
    lct_1d,
    learnable_lct_matrix,
)
from src.models.residual_lct_lstm import (
    ResidualAuxiliaryLCTRieszLSTMForecaster,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    ROOT / "experiments" / "numerical_validation" / "lct_riesz"
)
LENGTHS = (32, 60, 64, 128)
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEED = 20260825
FLOAT64_EQUIVALENCE_TOLERANCE = 1e-11
FLOAT64_ROUNDTRIP_TOLERANCE = 1e-11
FLOAT64_MULTIPLIER_TOLERANCE = 1e-14
PAPER_A_EQUALS_D_TOLERANCE = 1e-10


def _as_float(value: torch.Tensor | float) -> float:
    """Return a detached Python float for one scalar value."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _json_value(value: Any) -> Any:
    """Convert tensors, paths, and non-finite floats for JSON output."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write an indented UTF-8 JSON mapping."""
    path.write_text(
        json.dumps(_json_value(data), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write homogeneous validation rows to a UTF-8 CSV file."""
    if not rows:
        raise ValueError(f"Cannot write an empty validation table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return the L2 error relative to a nonzero expected tensor."""
    denominator = torch.linalg.vector_norm(expected)
    numerator = torch.linalg.vector_norm(actual - expected)
    if _as_float(denominator) == 0.0:
        return 0.0 if _as_float(numerator) == 0.0 else float("inf")
    return _as_float(numerator / denominator)


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Return elementwise and aggregate complex-safe error metrics."""
    error = (actual - expected).abs()
    return {
        "max_abs_error": _as_float(error.max()),
        "mae": _as_float(error.mean()),
        "rmse": _as_float(torch.sqrt(error.square().mean())),
        "relative_l2_error": _relative_l2(actual, expected),
    }


def _signals(length: int) -> dict[str, torch.Tensor]:
    """Create deterministic real validation signals of one requested length."""
    indices = torch.arange(length, dtype=torch.float64)
    impulse = torch.zeros(length, dtype=torch.float64)
    impulse[0] = 1.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + length)
    return {
        "unit_impulse": impulse,
        "single_sine": torch.sin(2.0 * torch.pi * 3.0 * indices / length),
        "dual_frequency": (
            0.8 * torch.cos(2.0 * torch.pi * 3.0 * indices / length)
            + 0.35 * torch.sin(2.0 * torch.pi * 7.0 * indices / length)
        ),
        "random_real": torch.randn(length, generator=generator, dtype=torch.float64),
        "chirp": torch.cos(
            2.0
            * torch.pi
            * (0.03 * indices + 0.22 * indices.square() / (2.0 * length))
        ),
    }


def _lct_matrix(alpha: float, m: float, q: float) -> tuple[torch.Tensor, ...]:
    """Construct an LCT matrix using float64 scalar parameters."""
    return learnable_lct_matrix(
        torch.tensor(alpha, dtype=torch.float64),
        torch.tensor(math.log(m), dtype=torch.float64),
        torch.tensor(q, dtype=torch.float64),
    )


def _complex_fit(reference: torch.Tensor, actual: torch.Tensor) -> complex:
    """Fit one global complex coefficient from reference to actual."""
    denominator = torch.vdot(reference, reference)
    if _as_float(denominator.real) == 0.0:
        return complex(1.0, 0.0)
    return complex((torch.vdot(reference, actual) / denominator).item())


def _frequency_order_candidates(spectrum: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return only theoretically common FFT frequency permutations."""
    length = spectrum.shape[-1]
    reversed_indices = torch.remainder(
        -torch.arange(length, device=spectrum.device),
        length,
    )
    reversed_spectrum = spectrum.index_select(-1, reversed_indices)
    return {
        "native": spectrum,
        "fftshift": torch.fft.fftshift(spectrum, dim=-1),
        "ifftshift": torch.fft.ifftshift(spectrum, dim=-1),
        "frequency_reversed": reversed_spectrum,
        "fftshift_reversed": torch.fft.fftshift(reversed_spectrum, dim=-1),
    }


def validate_fourier_equivalence() -> list[dict[str, Any]]:
    """Compare the Fourier LCT special case with PyTorch FFT conventions."""
    matrix = _lct_matrix(alpha=1.0, m=1.0, q=0.0)
    rows: list[dict[str, Any]] = []
    for length in LENGTHS:
        expected_coefficient = torch.polar(
            torch.tensor(1.0 / math.sqrt(length), dtype=torch.float64),
            torch.tensor(-math.pi / 4.0, dtype=torch.float64),
        ).to(torch.complex128)
        for signal_name, signal in _signals(length).items():
            actual = lct_1d(signal, matrix)
            fft_reference = torch.fft.fft(signal)
            raw = _error_metrics(actual, fft_reference)

            scale_coefficient = _as_float(
                torch.linalg.vector_norm(actual)
                / torch.linalg.vector_norm(fft_reference)
            )
            least_squares_coefficient = _complex_fit(fft_reference, actual)
            coefficient_tensor = torch.tensor(
                least_squares_coefficient,
                dtype=torch.complex128,
            )
            phase_coefficient = coefficient_tensor / coefficient_tensor.abs()
            scale_error = _error_metrics(
                actual,
                fft_reference * scale_coefficient,
            )
            phase_error = _error_metrics(actual, fft_reference * phase_coefficient)
            scale_phase_error = _error_metrics(
                actual,
                fft_reference * coefficient_tensor,
            )

            candidate_errors: list[tuple[float, int, str, str, complex]] = []
            conventions = {
                "fft_negative_exponent": torch.fft.fft(signal),
                "ifft_positive_exponent": torch.fft.ifft(signal),
            }
            convention_priority = {
                "fft_negative_exponent": 0,
                "ifft_positive_exponent": 10,
            }
            order_priority = {
                "native": 0,
                "frequency_reversed": 1,
                "fftshift": 2,
                "ifftshift": 3,
                "fftshift_reversed": 4,
            }
            for convention, reference in conventions.items():
                for order_name, ordered in _frequency_order_candidates(reference).items():
                    coefficient = _complex_fit(ordered, actual)
                    aligned = ordered * torch.tensor(
                        coefficient,
                        dtype=torch.complex128,
                    )
                    candidate_errors.append(
                        (
                            _relative_l2(actual, aligned),
                            convention_priority[convention] + order_priority[order_name],
                            convention,
                            order_name,
                            coefficient,
                        )
                    )
            numerical_minimum = min(item[0] for item in candidate_errors)
            tied_candidates = [
                item
                for item in candidate_errors
                if item[0] <= numerical_minimum + FLOAT64_EQUIVALENCE_TOLERANCE
            ]
            best_error, _, best_convention, best_order, best_coefficient = min(
                tied_candidates,
                key=lambda item: item[1],
            )
            best_reference = _frequency_order_candidates(
                conventions[best_convention]
            )[best_order]
            best_aligned = best_reference * torch.tensor(
                best_coefficient,
                dtype=torch.complex128,
            )
            best_metrics = _error_metrics(actual, best_aligned)
            coefficient_deviation = abs(
                least_squares_coefficient - complex(expected_coefficient.item())
            )

            rows.append(
                {
                    "length": length,
                    "signal": signal_name,
                    "raw_max_abs_error_vs_fft": raw["max_abs_error"],
                    "raw_relative_l2_error_vs_fft": raw["relative_l2_error"],
                    "scale_only_coefficient": scale_coefficient,
                    "scale_only_relative_l2_error": scale_error["relative_l2_error"],
                    "phase_only_rad": math.atan2(
                        least_squares_coefficient.imag,
                        least_squares_coefficient.real,
                    ),
                    "phase_only_relative_l2_error": phase_error["relative_l2_error"],
                    "global_coefficient_real": least_squares_coefficient.real,
                    "global_coefficient_imag": least_squares_coefficient.imag,
                    "global_scale": abs(least_squares_coefficient),
                    "global_phase_rad": math.atan2(
                        least_squares_coefficient.imag,
                        least_squares_coefficient.real,
                    ),
                    "global_scale_phase_max_abs_error": scale_phase_error[
                        "max_abs_error"
                    ],
                    "global_scale_phase_relative_l2_error": scale_phase_error[
                        "relative_l2_error"
                    ],
                    "theoretical_coefficient_real": _as_float(
                        expected_coefficient.real
                    ),
                    "theoretical_coefficient_imag": _as_float(
                        expected_coefficient.imag
                    ),
                    "coefficient_abs_deviation_from_exp_minus_i_pi_4_over_sqrt_n": (
                        coefficient_deviation
                    ),
                    "best_fft_or_ifft_convention": best_convention,
                    "best_frequency_order": best_order,
                    "best_global_scale": abs(best_coefficient),
                    "best_global_phase_rad": math.atan2(
                        best_coefficient.imag,
                        best_coefficient.real,
                    ),
                    "aligned_max_abs_error": best_metrics["max_abs_error"],
                    "aligned_relative_l2_error": best_error,
                    "only_global_scale": (
                        scale_error["relative_l2_error"]
                        <= FLOAT64_EQUIVALENCE_TOLERANCE
                    ),
                    "only_fixed_phase": (
                        phase_error["relative_l2_error"]
                        <= FLOAT64_EQUIVALENCE_TOLERANCE
                    ),
                    "only_global_scale_and_phase": (
                        scale_phase_error["relative_l2_error"]
                        <= FLOAT64_EQUIVALENCE_TOLERANCE
                    ),
                    "requires_fftshift": (
                        scale_phase_error["relative_l2_error"]
                        > FLOAT64_EQUIVALENCE_TOLERANCE
                        and "shift" in best_order
                    ),
                    "requires_frequency_reversal": (
                        scale_phase_error["relative_l2_error"]
                        > FLOAT64_EQUIVALENCE_TOLERANCE
                        and "reversed" in best_order
                    ),
                    "matches_fft_sign": best_convention == "fft_negative_exponent",
                }
            )
    return rows


def _roundtrip_parameter_sets(
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build requested synthetic and checkpoint-derived LCT parameter sets."""
    parameters: list[dict[str, Any]] = [
        {"source": "requested", "name": "fourier", "alpha": 1.0, "m": 1.0, "q": 0.0},
        *[
            {
                "source": "requested",
                "name": f"fractional_fourier_alpha_{alpha:g}",
                "alpha": alpha,
                "m": 1.0,
                "q": 0.0,
            }
            for alpha in (0.25, 0.5, 0.75)
        ],
        *[
            {
                "source": "requested",
                "name": f"scaling_chirp_m_{m:g}_q_{q:g}",
                "alpha": 0.5,
                "m": m,
                "q": q,
            }
            for m in (0.8, 1.2)
            for q in (-0.5, 0.5)
        ],
    ]
    for index, row in enumerate(inventory):
        parameters.append(
            {
                "source": "best_checkpoint",
                "name": f"checkpoint_{index:02d}_{Path(str(row['run_dir'])).name}",
                "run_dir": row["run_dir"],
                "alpha": float(row["alpha"]),
                "m": float(row["m"]),
                "q": float(row["q"]),
            }
        )
    return parameters


def validate_lct_roundtrip(
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Measure forward/inverse reconstruction for requested and learned LCTs."""
    rows: list[dict[str, Any]] = []
    for parameters in _roundtrip_parameter_sets(inventory):
        matrix = _lct_matrix(
            alpha=float(parameters["alpha"]),
            m=float(parameters["m"]),
            q=float(parameters["q"]),
        )
        b_value = _as_float(matrix[1])
        branch = "regular_B_nonzero" if abs(b_value) >= 1e-8 else "special_B_near_zero"
        for length in LENGTHS:
            for signal_name, signal in _signals(length).items():
                transformed = lct_1d(signal, matrix)
                reconstructed = lct_1d(
                    transformed,
                    inverse_lct_matrix(matrix),
                )
                metrics = _error_metrics(reconstructed, signal.to(torch.complex128))
                rows.append(
                    {
                        "source": parameters["source"],
                        "parameter_set": parameters["name"],
                        "run_dir": parameters.get("run_dir", ""),
                        "alpha": parameters["alpha"],
                        "m": parameters["m"],
                        "q": parameters["q"],
                        "B": b_value,
                        "implementation_branch": branch,
                        "length": length,
                        "signal": signal_name,
                        **metrics,
                        "passes_float64_threshold": (
                            metrics["relative_l2_error"]
                            <= FLOAT64_ROUNDTRIP_TOLERANCE
                        ),
                    }
                )
    return rows


def _parse_parameter_file(path: Path) -> dict[str, float]:
    """Parse the thesis-readable learned LCT parameter text file."""
    text = path.read_text(encoding="utf-8")
    patterns = {
        "alpha": r"^alpha\s*=\s*([^\s]+)",
        "m": r"^m\s*=\s*([^\s]+)",
        "q": r"^q\s*=\s*([^\s]+)",
        "gamma": r"^gamma\s*=\s*([^\s]+)",
        "residual_scale": r"^residual_scale\s*=\s*([^\s]+)",
    }
    values: dict[str, float] = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is not None:
            values[field] = float(match.group(1))
    matrix_match = re.search(
        r"\[ A\s+B \]\s*=\s*\[\s*([^\s]+)\s+([^\s]+)\s*\]"
        r"\s*\[ C\s+D \]\s*\[\s*([^\s]+)\s+([^\s]+)\s*\]",
        text,
        flags=re.MULTILINE,
    )
    if matrix_match is None:
        raise ValueError(f"Cannot parse LCT matrix from {path}")
    for field, value in zip(("A", "B", "C", "D"), matrix_match.groups()):
        values[field] = float(value)
    return values


def _checkpoint_for_parameter_file(path: Path) -> Path:
    """Map an experiment output directory to its best checkpoint path."""
    parts = list(path.parts)
    try:
        outputs_index = parts.index("outputs")
    except ValueError as error:
        raise ValueError(f"Parameter file is not below an outputs directory: {path}") from error
    parts[outputs_index] = "checkpoints"
    return Path(*parts[:-1]) / "best_model.pth"


def _load_checkpoint_scalars(path: Path) -> dict[str, float]:
    """Read LCT, gamma, gate, and residual scalars from a best checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    suffixes = {
        "alpha": "lct_riesz.lct.alpha_t",
        "log_m": "lct_riesz.lct.log_m_t",
        "q": "lct_riesz.lct.q_t",
        "gamma": "lct_riesz.gamma",
        "lct_gate": "lct_riesz.gate",
        "fusion_gate": "fusion_gate",
        "residual_scale": "residual_scale",
    }
    values: dict[str, float] = {}
    for field, suffix in suffixes.items():
        matching_keys = [key for key in state if key.endswith(suffix)]
        if len(matching_keys) == 1:
            values[field] = _as_float(state[matching_keys[0]])
        elif len(matching_keys) > 1:
            raise ValueError(f"Ambiguous checkpoint keys ending in {suffix}: {path}")
    if "log_m" in values:
        values["m"] = math.exp(values["log_m"])
    return values


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load one saved experiment config if it exists."""
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return {}
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _initial_values(config: Mapping[str, Any]) -> dict[str, float | bool]:
    """Resolve LCT initialization defaults exactly as the model factory does."""
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        model = {}
    model_type = str(model.get("type", "")).strip().lower()
    default_gate = 1.0 if model_type in {
        "dual_branch_lct_riesz_lstm",
        "residual_auxiliary_lct_riesz_lstm",
        "residual_auxiliary_signal_lstm",
    } else 0.0
    return {
        "alpha": float(model.get("lct_alpha", 1.0)),
        "m": float(model.get("lct_m", 1.0)),
        "q": float(model.get("lct_q", 0.0)),
        "gamma": float(model.get("riesz_gamma", 1.0)),
        "lct_gate": float(model.get("lct_gate_init", default_gate)),
        "residual_scale": float(model.get("residual_scale_init", 0.0)),
        "learnable_gamma": bool(model.get("learnable_gamma", False)),
    }


def _matrix_values(alpha: float, m: float, q: float) -> tuple[float, ...]:
    """Return detached float64 A, B, C, and D values."""
    return tuple(_as_float(value) for value in _lct_matrix(alpha, m, q))


def _matrix_distance(
    matrix: Sequence[float],
    reference: Sequence[float],
) -> float:
    """Return the Frobenius distance between two 2-by-2 matrices."""
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(matrix, reference)))


def _classify_learned_transform(
    *,
    alpha: float,
    m: float,
    q: float,
    matrix: Sequence[float],
    initial_matrix: Sequence[float],
) -> str:
    """Classify learned geometry using explicit quantitative thresholds."""
    if abs(matrix[1]) < 1e-6:
        return "B_near_zero_special_branch"
    change = math.sqrt((alpha - 1.0) ** 2 + (m - 1.0) ** 2 + q**2)
    if change < 0.02 and _matrix_distance(matrix, initial_matrix) < 0.04:
        return "almost_at_initial_fourier"
    fourier_distance = _matrix_distance(matrix, (0.0, 1.0, -1.0, 0.0))
    identity_distance = _matrix_distance(matrix, (1.0, 0.0, 0.0, 1.0))
    if identity_distance < fourier_distance:
        return "closer_to_identity"
    if abs(m - 1.0) <= 0.05 and abs(q) <= 0.05:
        return "fractional_fourier_near_fourier"
    return "fourier_near_with_chirp_or_scaling"


def scan_learned_parameter_inventory() -> list[dict[str, Any]]:
    """Inventory every real saved LCT run and its corresponding best checkpoint."""
    parameter_paths = sorted(
        (ROOT / "experiments").rglob("learned_lct_parameters.txt")
    )
    rows: list[dict[str, Any]] = []
    for parameter_path in parameter_paths:
        run_dir = parameter_path.parent
        text_values = _parse_parameter_file(parameter_path)
        checkpoint_path = _checkpoint_for_parameter_file(parameter_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing best checkpoint corresponding to {parameter_path}: {checkpoint_path}"
            )
        checkpoint_values = _load_checkpoint_scalars(checkpoint_path)
        config = _load_run_config(run_dir)
        initial = _initial_values(config)

        alpha = checkpoint_values.get("alpha", text_values["alpha"])
        m = checkpoint_values.get("m", text_values["m"])
        q = checkpoint_values.get("q", text_values["q"])
        gamma = checkpoint_values.get("gamma", text_values["gamma"])
        matrix = _matrix_values(alpha, m, q)
        initial_matrix = _matrix_values(
            float(initial["alpha"]),
            float(initial["m"]),
            float(initial["q"]),
        )
        determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2]
        fourier_distance = _matrix_distance(matrix, (0.0, 1.0, -1.0, 0.0))
        identity_distance = _matrix_distance(matrix, (1.0, 0.0, 0.0, 1.0))
        config_model = config.get("model", {}) if isinstance(config, Mapping) else {}
        model_type = (
            str(config_model.get("type", "lct_riesz_lstm"))
            if isinstance(config_model, Mapping)
            else "unknown"
        )
        text_matrix = tuple(text_values[field] for field in ("A", "B", "C", "D"))

        rows.append(
            {
                "run_dir": str(run_dir.relative_to(ROOT)),
                "model_type": model_type,
                "parameter_file": str(parameter_path.relative_to(ROOT)),
                "best_checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_epoch": int(
                    torch.load(
                        checkpoint_path,
                        map_location="cpu",
                        weights_only=True,
                    ).get("epoch", -1)
                ),
                "alpha": alpha,
                "m": m,
                "q": q,
                "A": matrix[0],
                "B": matrix[1],
                "C": matrix[2],
                "D": matrix[3],
                "determinant": determinant,
                "requires_A_equal_D": True,
                "abs_A_minus_D": abs(matrix[0] - matrix[3]),
                "satisfies_paper_multiplier_theorem_conditions": (
                    abs(matrix[0] - matrix[3]) <= PAPER_A_EQUALS_D_TOLERANCE
                    and abs(matrix[1]) >= 1e-6
                ),
                "gamma": gamma,
                "gamma_learnable": initial["learnable_gamma"],
                "lct_gate": checkpoint_values.get("lct_gate", float("nan")),
                "fusion_gate": checkpoint_values.get("fusion_gate", float("nan")),
                "residual_scale": checkpoint_values.get(
                    "residual_scale",
                    text_values.get("residual_scale", float("nan")),
                ),
                "initial_alpha": initial["alpha"],
                "initial_m": initial["m"],
                "initial_q": initial["q"],
                "initial_gamma": initial["gamma"],
                "initial_lct_gate": initial["lct_gate"],
                "initial_residual_scale": initial["residual_scale"],
                "delta_alpha": alpha - float(initial["alpha"]),
                "delta_m": m - float(initial["m"]),
                "delta_q": q - float(initial["q"]),
                "delta_gamma": gamma - float(initial["gamma"]),
                "delta_lct_gate": checkpoint_values.get(
                    "lct_gate",
                    float("nan"),
                )
                - float(initial["lct_gate"]),
                "delta_residual_scale": checkpoint_values.get(
                    "residual_scale",
                    text_values.get("residual_scale", float("nan")),
                )
                - float(initial["residual_scale"]),
                "matrix_distance_from_initial": _matrix_distance(
                    matrix,
                    initial_matrix,
                ),
                "fourier_matrix_distance_d_F": fourier_distance,
                "identity_matrix_distance_d_I": identity_distance,
                "closer_to_fourier_than_identity": fourier_distance < identity_distance,
                "parameterization_distance_from_fractional_rotation": math.sqrt(
                    (m - 1.0) ** 2 + q**2
                ),
                "B_near_zero": abs(matrix[1]) < 1e-6,
                "classification": _classify_learned_transform(
                    alpha=alpha,
                    m=m,
                    q=q,
                    matrix=matrix,
                    initial_matrix=initial_matrix,
                ),
                "max_abs_text_vs_checkpoint_matrix_difference": max(
                    abs(left - right) for left, right in zip(text_matrix, matrix)
                ),
                "abs_text_vs_checkpoint_alpha_difference": abs(
                    text_values["alpha"] - alpha
                ),
                "abs_text_vs_checkpoint_m_difference": abs(text_values["m"] - m),
                "abs_text_vs_checkpoint_q_difference": abs(text_values["q"] - q),
            }
        )
    if not rows:
        raise FileNotFoundError("No learned_lct_parameters.txt files were found.")
    return rows


def validate_riesz_multiplier() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check the project's fractional Hilbert interpolation extension formula."""
    multiplier_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    for length in LENGTHS:
        frequencies = torch.fft.fftfreq(length, dtype=torch.float64)
        signs = torch.sign(frequencies)
        for gamma in GAMMAS:
            code = fractional_riesz_multiplier(
                length,
                gamma,
                device=torch.device("cpu"),
                dtype=torch.complex128,
            )
            phase = torch.tensor(gamma * math.pi / 2.0, dtype=torch.float64)
            extension_formula = torch.cos(phase) - 1j * signs * torch.sin(phase)
            extension_formula = extension_formula.to(torch.complex128)
            for index in range(length):
                is_dc = index == 0
                is_nyquist = length % 2 == 0 and index == length // 2
                multiplier_rows.append(
                    {
                        "length": length,
                        "gamma": gamma,
                        "bin": index,
                        "frequency_cycles_per_sample": _as_float(frequencies[index]),
                        "frequency_sign": _as_float(signs[index]),
                        "is_dc": is_dc,
                        "is_even_length_nyquist": is_nyquist,
                        "code_real": _as_float(code[index].real),
                        "code_imag": _as_float(code[index].imag),
                        "project_extension_formula_real": _as_float(
                            extension_formula[index].real
                        ),
                        "project_extension_formula_imag": _as_float(
                            extension_formula[index].imag
                        ),
                        "complex_abs_error": _as_float(
                            (code[index] - extension_formula[index]).abs()
                        ),
                        "code_magnitude": _as_float(code[index].abs()),
                        "project_extension_formula_magnitude": _as_float(
                            extension_formula[index].abs()
                        ),
                    }
                )

        response_signals = {
            key: value
            for key, value in _signals(length).items()
            if key in {"single_sine", "random_real"}
        }
        indices = torch.arange(length, dtype=torch.float64)
        response_signals["single_cosine"] = torch.cos(
            2.0 * torch.pi * 3.0 * indices / length
        )
        previous_outputs: dict[str, torch.Tensor] = {}
        for gamma in GAMMAS:
            multiplier = fractional_riesz_multiplier(
                length,
                gamma,
                device=torch.device("cpu"),
                dtype=torch.complex128,
            )
            for signal_name, signal in response_signals.items():
                output = torch.fft.ifft(torch.fft.fft(signal) * multiplier)
                previous = previous_outputs.get(signal_name)
                continuity = (
                    float("nan")
                    if previous is None
                    else _relative_l2(output, previous)
                )
                previous_outputs[signal_name] = output
                expected_wave_error = float("nan")
                if signal_name in {"single_sine", "single_cosine"}:
                    base_phase = 0.0 if signal_name == "single_cosine" else -math.pi / 2.0
                    expected = torch.cos(
                        2.0 * torch.pi * 3.0 * indices / length
                        + base_phase
                        - gamma * math.pi / 2.0
                    )
                    expected_wave_error = _relative_l2(output.real, expected)
                response_rows.append(
                    {
                        "length": length,
                        "gamma": gamma,
                        "signal": signal_name,
                        "input_mean": _as_float(signal.mean()),
                        "output_real_mean": _as_float(output.real.mean()),
                        "output_real_l2": _as_float(torch.linalg.vector_norm(output.real)),
                        "output_imag_l2": _as_float(torch.linalg.vector_norm(output.imag)),
                        "relative_change_from_previous_gamma": continuity,
                        "relative_error_vs_phase_shifted_wave": expected_wave_error,
                    }
                )
    return multiplier_rows, response_rows


def _gradient_record(parameter: torch.Tensor) -> dict[str, Any]:
    """Describe whether one scalar gradient exists, is finite, and is nonzero."""
    gradient = parameter.grad
    if gradient is None:
        return {
            "is_none": True,
            "is_finite": False,
            "absolute_value": None,
            "near_numerical_zero": None,
        }
    absolute_value = _as_float(gradient.detach().abs().max())
    return {
        "is_none": False,
        "is_finite": bool(torch.isfinite(gradient).all().item()),
        "absolute_value": absolute_value,
        "near_numerical_zero": absolute_value < 1e-12,
    }


def _core_gradient_diagnostic(gate_init: float) -> dict[str, Any]:
    """Backpropagate once through the standalone LCT-Riesz block."""
    torch.manual_seed(SEED)
    module = LearnableLCTRiesz1D(
        channels=3,
        alpha=0.73,
        m=1.08,
        q=0.17,
        gamma=0.6,
        learnable_gamma=True,
        gate_init=gate_init,
    ).double()
    inputs = torch.randn(4, 3, 60, dtype=torch.float64)
    weights = torch.linspace(0.25, 1.25, 60, dtype=torch.float64).view(1, 1, -1)
    output = module(inputs)
    loss = (output.square() * weights).mean() + 0.01 * output.mean()
    loss.backward()
    with torch.no_grad():
        original_q = module.lct.q_t.detach().clone()
        q_outputs = []
        for q_value in (-0.5, 0.0, 0.5):
            module.lct.q_t.fill_(q_value)
            q_outputs.append(module(inputs).detach().clone())
        module.lct.q_t.copy_(original_q)
    q_output_max_change = max(
        _as_float((left - right).abs().max())
        for left in q_outputs
        for right in q_outputs
    )
    return {
        "description": "standalone LearnableLCTRiesz1D forward/backward",
        "sequence_length": 60,
        "gate_initial_value": gate_init,
        "loss": _as_float(loss),
        "q_output_max_abs_change_for_q_minus_0_5_0_plus_0_5": q_output_max_change,
        "alpha_t": _gradient_record(module.lct.alpha_t),
        "log_m_t": _gradient_record(module.lct.log_m_t),
        "q_t": _gradient_record(module.lct.q_t),
        "learnable_gamma": _gradient_record(module.gamma),
        "lct_gate": _gradient_record(module.gate),
    }


def _residual_gradient_diagnostic(residual_scale_init: float) -> dict[str, Any]:
    """Backpropagate once through the formal residual forecasting architecture."""
    torch.manual_seed(SEED)
    model = ResidualAuxiliaryLCTRieszLSTMForecaster(
        input_dim=5,
        hidden_dim=8,
        lstm_hidden_dim=12,
        signal_feature_indices=(1, 2, 3),
        num_layers=1,
        output_dim=1,
        dropout=0.0,
        use_lct_riesz=True,
        spectral_hidden_dim=8,
        residual_scale_init=residual_scale_init,
        lct_alpha=0.73,
        lct_m=1.08,
        lct_q=0.17,
        riesz_gamma=0.6,
        learnable_gamma=True,
        lct_gate_init=1.0,
    ).double()
    inputs = torch.randn(4, 60, 5, dtype=torch.float64)
    targets = torch.linspace(-0.2, 0.3, 4, dtype=torch.float64).view(-1, 1)
    prediction = model(inputs)
    loss = torch.mean((prediction - targets).square())
    loss.backward()
    lct_module = model.lct_riesz
    if not isinstance(lct_module, LearnableLCTRiesz1D):
        raise TypeError("Residual diagnostic did not construct LearnableLCTRiesz1D.")
    return {
        "description": "residual forecaster final prediction forward/backward",
        "sequence_length": 60,
        "residual_scale_initial_value": residual_scale_init,
        "loss": _as_float(loss),
        "alpha_t": _gradient_record(lct_module.lct.alpha_t),
        "log_m_t": _gradient_record(lct_module.lct.log_m_t),
        "q_t": _gradient_record(lct_module.lct.q_t),
        "learnable_gamma": _gradient_record(lct_module.gamma),
        "lct_gate": _gradient_record(lct_module.gate),
        "residual_scale": _gradient_record(model.residual_scale),
    }


def validate_gradients() -> dict[str, Any]:
    """Diagnose open and intentionally gated gradient paths without an update."""
    return {
        "dtype": "float64",
        "near_zero_threshold": 1e-12,
        "optimizer_step_performed": False,
        "standalone_open_gate": _core_gradient_diagnostic(gate_init=1.0),
        "standalone_zero_gate": _core_gradient_diagnostic(gate_init=0.0),
        "residual_zero_scale": _residual_gradient_diagnostic(
            residual_scale_init=0.0
        ),
        "residual_nonzero_scale_control": _residual_gradient_diagnostic(
            residual_scale_init=0.01
        ),
    }


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save one validation figure as 300 DPI PNG and PDF."""
    figure.tight_layout()
    try:
        figure.savefig(
            output_dir / f"{stem}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        figure.savefig(
            output_dir / f"{stem}.pdf",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    finally:
        plt.close(figure)


def plot_fourier_errors(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Plot raw and convention-aligned Fourier relative errors."""
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for signal_name in sorted({str(row["signal"]) for row in rows}):
        selected = [row for row in rows if row["signal"] == signal_name]
        axis.plot(
            [int(row["length"]) for row in selected],
            [float(row["aligned_relative_l2_error"]) for row in selected],
            marker="o",
            linewidth=1.4,
            label=f"{signal_name}: aligned",
        )
    raw_max = [float(row["raw_relative_l2_error_vs_fft"]) for row in rows]
    axis.axhline(min(raw_max), color="black", linestyle="--", linewidth=1.0, label="minimum raw error")
    axis.set_yscale("log")
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Relative L2 error")
    axis.set_title("Fourier LCT Equivalence After Global Convention Alignment")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    _save_figure(figure, output_dir, "fourier_equivalence_error")


def plot_roundtrip_errors(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Plot worst LCT roundtrip relative error by length and source."""
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    sources = sorted({str(row["source"]) for row in rows})
    for source in sources:
        length_errors = []
        for length in LENGTHS:
            values = [
                float(row["relative_l2_error"])
                for row in rows
                if row["source"] == source and int(row["length"]) == length
            ]
            length_errors.append(max(values))
        axis.plot(LENGTHS, length_errors, marker="o", linewidth=1.6, label=source)
    axis.axhline(
        FLOAT64_ROUNDTRIP_TOLERANCE,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="float64 threshold",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Worst relative L2 error")
    axis.set_title("LCT Forward/Inverse Reconstruction")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    _save_figure(figure, output_dir, "lct_roundtrip_error")


def plot_riesz_validation(
    multiplier_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    """Plot errors against the project's Hilbert interpolation extension formula."""
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    full_errors = []
    non_dc_errors = []
    for gamma in GAMMAS:
        selected = [row for row in multiplier_rows if float(row["gamma"]) == gamma]
        full_errors.append(max(float(row["complex_abs_error"]) for row in selected))
        non_dc_errors.append(
            max(
                float(row["complex_abs_error"])
                for row in selected
                if not bool(row["is_dc"])
            )
        )
    plot_floor = 1e-18
    axis.plot(
        GAMMAS,
        [max(value, plot_floor) for value in full_errors],
        marker="o",
        linewidth=1.6,
        label="all bins",
    )
    axis.plot(
        GAMMAS,
        [max(value, plot_floor) for value in non_dc_errors],
        marker="s",
        linewidth=1.6,
        label="excluding DC (exact zero plotted at 1e-18)",
    )
    axis.axhline(
        FLOAT64_MULTIPLIER_TOLERANCE,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="float64 threshold",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Project Hilbert interpolation order gamma")
    axis.set_ylabel("Maximum complex absolute error")
    axis.set_title("Project Fractional Hilbert Multiplier Formula Check")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    _save_figure(figure, output_dir, "riesz_multiplier_error")


def _git_value(arguments: Sequence[str]) -> str:
    """Return one read-only Git value for validation provenance."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _max_value(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    """Return the maximum numeric field value across validation rows."""
    return max(float(row[field]) for row in rows)


def build_summary(
    *,
    output_dir: Path,
    fourier_rows: Sequence[Mapping[str, Any]],
    roundtrip_rows: Sequence[Mapping[str, Any]],
    multiplier_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
    inventory_rows: Sequence[Mapping[str, Any]],
    gradients: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the final evidence-backed numerical validation summary."""
    fourier_max_aligned = _max_value(fourier_rows, "aligned_relative_l2_error")
    roundtrip_max = _max_value(roundtrip_rows, "relative_l2_error")
    roundtrip_by_length = {
        str(length): max(
            float(row["relative_l2_error"])
            for row in roundtrip_rows
            if int(row["length"]) == length
        )
        for length in LENGTHS
    }
    extension_formula_full_max = _max_value(multiplier_rows, "complex_abs_error")
    extension_formula_non_dc_max = max(
        float(row["complex_abs_error"])
        for row in multiplier_rows
        if not bool(row["is_dc"])
    )
    dc_rows = [row for row in multiplier_rows if bool(row["is_dc"])]
    nyquist_rows = [
        row for row in multiplier_rows if bool(row["is_even_length_nyquist"])
    ]
    gamma_zero_dc = [row for row in dc_rows if float(row["gamma"]) == 0.0]
    gamma_one_nyquist = [
        row for row in nyquist_rows if float(row["gamma"]) == 1.0
    ]
    open_gate_gradients = gradients["standalone_open_gate"]
    active_gradient_fields = ("alpha_t", "log_m_t", "learnable_gamma", "lct_gate")
    active_gradients_reach = all(
        not bool(open_gate_gradients[field]["is_none"])
        and bool(open_gate_gradients[field]["is_finite"])
        and not bool(open_gate_gradients[field]["near_numerical_zero"])
        for field in active_gradient_fields
    )
    q_gradient_is_structurally_inactive = bool(
        open_gate_gradients["q_t"]["near_numerical_zero"]
    )
    fourier_pass = fourier_max_aligned <= FLOAT64_EQUIVALENCE_TOLERANCE
    roundtrip_pass = roundtrip_max <= FLOAT64_ROUNDTRIP_TOLERANCE
    extension_formula_full_pass = (
        extension_formula_full_max <= FLOAT64_MULTIPLIER_TOLERANCE
    )
    extension_formula_non_dc_pass = (
        extension_formula_non_dc_max <= FLOAT64_MULTIPLIER_TOLERANCE
    )
    determinant_max_abs_error = max(
        abs(float(row["determinant"]) - 1.0) for row in inventory_rows
    )
    determinant_pass = determinant_max_abs_error <= FLOAT64_EQUIVALENCE_TOLERANCE
    core_gradients_reach = all(
        not bool(open_gate_gradients[field]["is_none"])
        and bool(open_gate_gradients[field]["is_finite"])
        and not bool(open_gate_gradients[field]["near_numerical_zero"])
        for field in ("alpha_t", "log_m_t")
    )
    discrete_lct_core_pass = (
        fourier_pass and roundtrip_pass and determinant_pass and core_gradients_reach
    )
    paper_multiplier_condition_count = sum(
        bool(row["satisfies_paper_multiplier_theorem_conditions"])
        for row in inventory_rows
    )
    all_formal_experiments_gamma_one = all(
        abs(float(row["gamma"]) - 1.0) <= FLOAT64_MULTIPLIER_TOLERANCE
        for row in inventory_rows
    )

    categories: dict[str, int] = {}
    for row in inventory_rows:
        category = str(row["classification"])
        categories[category] = categories.get(category, 0) + 1

    return {
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "output_dir": str(output_dir.relative_to(ROOT)),
            "git_branch": _git_value(("branch", "--show-current")),
            "git_commit": _git_value(("rev-parse", "HEAD")),
            "torch_version": torch.__version__,
            "device": "cpu",
            "real_dtype": "float64",
            "complex_dtype": "complex128",
            "seed": SEED,
            "training_performed": False,
            "optimizer_step_performed": False,
        },
        "thresholds": {
            "fourier_relative_l2": FLOAT64_EQUIVALENCE_TOLERANCE,
            "roundtrip_relative_l2": FLOAT64_ROUNDTRIP_TOLERANCE,
            "riesz_complex_abs": FLOAT64_MULTIPLIER_TOLERANCE,
            "rationale": (
                "All core comparisons use float64/complex128. Exact FFT/chirp identities "
                "should be within a modest multiple of double-precision FFT rounding; "
                "1e-11 allows length-dependent accumulation without masking percent-scale errors."
            ),
        },
        "implementation": {
            "lct_type": "same-length circular chirp-FFT-IFFT-chirp discrete approximation",
            "forward": "lct_1d(signal, matrix)",
            "inverse": "lct_1d(signal, inverse_lct_matrix(matrix))",
            "complex_domain_access": "LearnableLCT1D.forward or LearnableLCTRiesz1D.lct",
            "fft_normalization": "torch default backward: FFT unnormalized, IFFT has 1/N",
            "fft_sign": "FFT negative exponent; IFFT positive exponent",
            "fftshift_used_in_production": False,
            "lct_grid": "n=0,...,N-1 with chirp exp(i*pi*theta*n^2/N)",
            "riesz_grid": "torch.fft.fftfreq(N) in native FFT ordering",
            "project_fractional_hilbert_extension_formula": (
                "[cos(pi*gamma/2)-i*sign(f)*sin(pi*gamma/2)] * 1[f!=0]"
            ),
            "paper_lcrt_multiplier_theorem_requires_A_equal_D": True,
            "paper_lcrt_one_dimensional_multiplier": "-i*sign(omega/B), omega!=0",
        },
        "fourier_equivalence": {
            "tested_cases": len(fourier_rows),
            "max_raw_relative_l2_error_vs_fft": _max_value(
                fourier_rows,
                "raw_relative_l2_error_vs_fft",
            ),
            "max_raw_absolute_error_vs_fft": _max_value(
                fourier_rows,
                "raw_max_abs_error_vs_fft",
            ),
            "max_aligned_absolute_error": _max_value(
                fourier_rows,
                "aligned_max_abs_error",
            ),
            "max_aligned_relative_l2_error": fourier_max_aligned,
            "all_best_match_fft_native": all(
                row["best_fft_or_ifft_convention"] == "fft_negative_exponent"
                and row["best_frequency_order"] == "native"
                for row in fourier_rows
            ),
            "theoretical_relation": "LCT_Fourier(x)=exp(-i*pi/4)*FFT(x)/sqrt(N)",
            "only_global_scale_for_all_cases": all(
                bool(row["only_global_scale"]) for row in fourier_rows
            ),
            "only_fixed_phase_for_all_cases": all(
                bool(row["only_fixed_phase"]) for row in fourier_rows
            ),
            "only_one_global_scale_and_phase_for_all_cases": all(
                bool(row["only_global_scale_and_phase"]) for row in fourier_rows
            ),
            "requires_fftshift": any(bool(row["requires_fftshift"]) for row in fourier_rows),
            "requires_frequency_reversal": any(
                bool(row["requires_frequency_reversal"]) for row in fourier_rows
            ),
            "passes": fourier_pass,
        },
        "lct_roundtrip": {
            "tested_cases": len(roundtrip_rows),
            "max_absolute_error": _max_value(roundtrip_rows, "max_abs_error"),
            "max_mae": _max_value(roundtrip_rows, "mae"),
            "max_rmse": _max_value(roundtrip_rows, "rmse"),
            "max_relative_l2_error": roundtrip_max,
            "worst_relative_l2_by_length": roundtrip_by_length,
            "length_60_special_problem": (
                roundtrip_by_length["60"]
                > 10.0 * min(roundtrip_by_length.values())
                and roundtrip_by_length["60"] > FLOAT64_ROUNDTRIP_TOLERANCE
            ),
            "B_near_zero_cases": sum(
                row["implementation_branch"] == "special_B_near_zero"
                for row in roundtrip_rows
            ),
            "passes": roundtrip_pass,
        },
        "project_fractional_hilbert_extension": {
            "provenance": (
                "Project-defined interpolation formula; it is not the fractional "
                "Riesz definition or multiplier theorem of the audited LCRT paper."
            ),
            "tested_multiplier_bins": len(multiplier_rows),
            "max_complex_abs_error_vs_project_formula_all_bins": (
                extension_formula_full_max
            ),
            "max_complex_abs_error_vs_project_formula_excluding_dc": (
                extension_formula_non_dc_max
            ),
            "dc_code_values": sorted(
                {complex(float(row["code_real"]), float(row["code_imag"])).__repr__() for row in dc_rows}
            ),
            "gamma_zero_is_identity": all(
                abs(float(row["code_real"]) - 1.0) <= FLOAT64_MULTIPLIER_TOLERANCE
                and abs(float(row["code_imag"])) <= FLOAT64_MULTIPLIER_TOLERANCE
                for row in multiplier_rows
                if float(row["gamma"]) == 0.0
            ),
            "gamma_zero_dc_abs_error": max(
                float(row["complex_abs_error"]) for row in gamma_zero_dc
            ),
            "dc_interpretation": (
                "The unconditional DC mask makes the project's gamma=0 endpoint "
                "mean-removing rather than Identity. DC=0 at gamma=1 is consistent "
                "with the classical discrete Hilbert convention and is not an error "
                "in the paper's gamma=1 LCRT semantics."
            ),
            "all_formal_experiments_use_gamma_one": all_formal_experiments_gamma_one,
            "historical_gamma_one_experiments_invalidated_by_dc": False,
            "gamma_one_hilbert_non_dc": all(
                float(row["complex_abs_error"]) <= FLOAT64_MULTIPLIER_TOLERANCE
                for row in multiplier_rows
                if float(row["gamma"]) == 1.0 and not bool(row["is_dc"])
            ),
            "even_nyquist_gamma_one_values": sorted(
                {
                    complex(float(row["code_real"]), float(row["code_imag"])).__repr__()
                    for row in gamma_one_nyquist
                }
            ),
            "nyquist_convention": (
                "Even-length fftfreq labels Nyquist as negative, so gamma=1 gives +i; "
                "the implementation does not apply the common real-Hilbert Nyquist zero."
            ),
            "non_dc_multiplier_magnitude_min": min(
                float(row["code_magnitude"])
                for row in multiplier_rows
                if not bool(row["is_dc"])
            ),
            "non_dc_multiplier_magnitude_max": max(
                float(row["code_magnitude"])
                for row in multiplier_rows
                if not bool(row["is_dc"])
            ),
            "time_response_cases": len(response_rows),
            "successive_gamma_response_relative_change_range": [
                min(
                    float(row["relative_change_from_previous_gamma"])
                    for row in response_rows
                    if math.isfinite(float(row["relative_change_from_previous_gamma"]))
                ),
                max(
                    float(row["relative_change_from_previous_gamma"])
                    for row in response_rows
                    if math.isfinite(float(row["relative_change_from_previous_gamma"]))
                ),
            ],
            "max_random_signal_imaginary_l2_from_nyquist": max(
                float(row["output_imag_l2"])
                for row in response_rows
                if row["signal"] == "random_real"
            ),
            "max_sinusoid_phase_shift_relative_error": max(
                float(row["relative_error_vs_phase_shifted_wave"])
                for row in response_rows
                if row["signal"] in {"single_sine", "single_cosine"}
            ),
            "passes_project_extension_formula_all_bins": (
                extension_formula_full_pass
            ),
            "passes_project_extension_formula_excluding_dc": (
                extension_formula_non_dc_pass
            ),
        },
        "learned_parameter_inventory": {
            "run_count": len(inventory_rows),
            "B_near_zero_count": sum(bool(row["B_near_zero"]) for row in inventory_rows),
            "paper_multiplier_theorem_requires_A_equal_D": True,
            "A_equals_D_tolerance": PAPER_A_EQUALS_D_TOLERANCE,
            "paper_multiplier_theorem_condition_count": (
                paper_multiplier_condition_count
            ),
            "all_checkpoints_satisfy_paper_multiplier_theorem_conditions": (
                paper_multiplier_condition_count == len(inventory_rows)
            ),
            "abs_A_minus_D_range": [
                min(float(row["abs_A_minus_D"]) for row in inventory_rows),
                max(float(row["abs_A_minus_D"]) for row in inventory_rows),
            ],
            "determinant_max_abs_error_from_one": determinant_max_abs_error,
            "all_closer_to_fourier_than_identity": all(
                bool(row["closer_to_fourier_than_identity"])
                for row in inventory_rows
            ),
            "classification_counts": categories,
            "max_fourier_distance": max(
                float(row["fourier_matrix_distance_d_F"])
                for row in inventory_rows
            ),
            "max_text_checkpoint_matrix_difference": max(
                float(row["max_abs_text_vs_checkpoint_matrix_difference"])
                for row in inventory_rows
            ),
            "ranges": {
                field: [
                    min(float(row[field]) for row in inventory_rows),
                    max(float(row[field]) for row in inventory_rows),
                ]
                for field in (
                    "alpha",
                    "m",
                    "q",
                    "A",
                    "B",
                    "C",
                    "D",
                    "determinant",
                    "abs_A_minus_D",
                    "gamma",
                    "lct_gate",
                    "fourier_matrix_distance_d_F",
                )
            },
        },
        "gradient_diagnostics": {
            "standalone_open_gate_active_gradients_finite_nonzero": (
                active_gradients_reach
            ),
            "standalone_open_gate_q_is_numerically_inactive": bool(
                gradients["standalone_open_gate"]["q_t"]["near_numerical_zero"]
            ),
            "q_interpretation": (
                "q cancels algebraically in the current LCT-multiplier-inverse-LCT "
                "conjugation, so its near-zero gradient is structural parameter "
                "non-identifiability rather than an LCT forward/inverse error."
            ),
            "standalone_zero_gate_blocks_transform_parameter_gradients": all(
                bool(gradients["standalone_zero_gate"][field]["near_numerical_zero"])
                for field in ("alpha_t", "log_m_t", "q_t", "learnable_gamma")
            ),
            "residual_zero_scale_blocks_spectral_gradients": all(
                bool(gradients["residual_zero_scale"][field]["near_numerical_zero"])
                for field in ("alpha_t", "log_m_t", "q_t", "learnable_gamma", "lct_gate")
            ),
            "residual_scale_gradient_at_zero_is_nonzero": not bool(
                gradients["residual_zero_scale"]["residual_scale"]["near_numerical_zero"]
            ),
            "residual_nonzero_scale_reopens_lct_gradients": all(
                not bool(gradients["residual_nonzero_scale_control"][field]["near_numerical_zero"])
                for field in ("alpha_t", "log_m_t", "q_t", "learnable_gamma", "lct_gate")
            ),
            "residual_nonzero_scale_reopens_active_gradients_except_q": all(
                not bool(gradients["residual_nonzero_scale_control"][field]["near_numerical_zero"])
                for field in ("alpha_t", "log_m_t", "learnable_gamma", "lct_gate")
            ),
        },
        "layered_classification": {
            "discrete_lct_core": {
                "category": "A" if discrete_lct_core_pass else "C",
                "label": (
                    "数值实现正确"
                    if discrete_lct_core_pass
                    else "离散LCT核心验证未全部通过"
                ),
                "evidence": (
                    "Fourier special-case equivalence, forward/inverse roundtrip, "
                    "unit determinant, and active alpha/m gradients are evaluated "
                    "independently of the upper filtering interpretation."
                ),
            },
            "original_paper_lcrt_correspondence": {
                "category": (
                    "A"
                    if paper_multiplier_condition_count == len(inventory_rows)
                    else "C"
                ),
                "label": "当前通用模块不满足严格原论文LCRT对应条件",
                "reason": (
                    "The paper's simple LCT-domain multiplier theorem requires A=D "
                    "and B!=0. The learnable parameterization enforces determinant one "
                    "but does not enforce A=D; outside that condition the current "
                    "conjugated multiplier cannot be claimed equivalent to Definition 2.5."
                ),
            },
            "project_gamma_extension": {
                "category": "NEEDS_REVISION",
                "label": "项目fractional Hilbert interpolation扩展需要修订",
                "reason": (
                    "The formula is a project extension. Its unconditional DC zeroing "
                    "breaks gamma=0=Identity, while gamma=1 DC zero is a conventional "
                    "discrete Hilbert choice. The even-length Nyquist convention must "
                    "be stated explicitly."
                ),
                "historical_gamma_one_experiments_remain_empirically_valid": (
                    all_formal_experiments_gamma_one
                ),
            },
            "q_parameter": {
                "category": "STRUCTURAL_REDUNDANCY",
                "label": "当前共轭滤波结构中的不可辨识参数",
                "confirmed_by_near_zero_gradient": q_gradient_is_structurally_inactive,
                "reason": (
                    "The q-dependent output chirp commutes with the pointwise domain "
                    "multiplier and cancels against the inverse LCT. Historical q values "
                    "must not be interpreted as learned transform contributions."
                ),
            },
            "recommended_research_sequence": (
                "Establish a strict paper-LCRT baseline first, then define and ablate "
                "the gamma extension. No production-model correction is performed here."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse the optional timestamped output directory override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output directory; default uses a local timestamp.",
    )
    return parser.parse_args()


def main() -> Path:
    """Run every diagnostic and return the timestamped artifact directory."""
    args = parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    inventory_rows = scan_learned_parameter_inventory()
    fourier_rows = validate_fourier_equivalence()
    roundtrip_rows = validate_lct_roundtrip(inventory_rows)
    multiplier_rows, response_rows = validate_riesz_multiplier()
    gradients = validate_gradients()

    _write_csv(output_dir / "fourier_equivalence.csv", fourier_rows)
    _write_csv(output_dir / "lct_roundtrip.csv", roundtrip_rows)
    _write_csv(
        output_dir / "riesz_multiplier_validation.csv",
        multiplier_rows,
    )
    _write_csv(output_dir / "riesz_time_response.csv", response_rows)
    _write_csv(
        output_dir / "learned_lct_parameter_inventory.csv",
        inventory_rows,
    )
    _write_json(output_dir / "gradient_diagnostics.json", gradients)

    plot_fourier_errors(fourier_rows, output_dir)
    plot_roundtrip_errors(roundtrip_rows, output_dir)
    plot_riesz_validation(multiplier_rows, output_dir)

    summary = build_summary(
        output_dir=output_dir,
        fourier_rows=fourier_rows,
        roundtrip_rows=roundtrip_rows,
        multiplier_rows=multiplier_rows,
        response_rows=response_rows,
        inventory_rows=inventory_rows,
        gradients=gradients,
    )
    _write_json(output_dir / "validation_summary.json", summary)
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
