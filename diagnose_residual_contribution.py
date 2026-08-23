"""Diagnose a trained residual LCT-Riesz branch without updating weights."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate import load_checkpoint
from src.data.dataset import FinancialDataLoaders, create_dataloaders
from src.utils.metrics import calculate_all_metrics
from train import build_model, load_config, resolve_device, set_seed


PathLike = Union[str, Path]
METRIC_NAMES = ("mae", "mse", "rmse", "r2")
DEFAULT_CONFIG = Path(
    "experiments/lstm/outputs/volatility_5/residual_lct_signal_features/"
    "20260625_163554/config.json"
)
DEFAULT_CHECKPOINT = Path(
    "experiments/lstm/checkpoints/volatility_5/residual_lct_signal_features/"
    "20260625_163554/best_model.pth"
)
DEFAULT_ORIGINAL_RUN = Path(
    "experiments/lstm/outputs/volatility_5/residual_lct_signal_features/"
    "20260625_163554"
)
DEFAULT_ORIGINAL_METRICS = DEFAULT_ORIGINAL_RUN / "metrics.json"
DEFAULT_ORIGINAL_PREDICTIONS = DEFAULT_ORIGINAL_RUN / "prediction_results.csv"
DEFAULT_OUTPUT_ROOT = Path(
    "experiments/lstm/diagnostics/residual_contribution"
)


def _flat_batch_tensor(
    value: torch.Tensor,
    name: str,
    batch_size: int,
) -> np.ndarray:
    """Convert a ``[batch]`` or ``[batch, 1]`` tensor to float64 values."""
    if value.shape not in {(batch_size,), (batch_size, 1)}:
        raise ValueError(
            "{} must have shape [batch] or [batch, 1], received {}.".format(
                name,
                tuple(value.shape),
            )
        )
    array = value.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError("{} contains NaN or Inf.".format(name))
    return array


def _restore_evaluation_scale(
    values: Dict[str, np.ndarray],
    target_scaler: Any,
) -> Dict[str, np.ndarray]:
    """Restore target-scale values while preserving residual decomposition."""
    if target_scaler is None:
        return values

    scale = float(np.asarray(target_scaler.scale_, dtype=np.float64).reshape(-1)[0])
    restored = dict(values)
    for name in ("target", "main_pred", "final_pred"):
        restored[name] = target_scaler.inverse_transform(
            values[name].reshape(-1, 1)
        ).reshape(-1)
    restored["spectral_delta"] = values["spectral_delta"] * scale
    restored["residual_correction"] = values["residual_correction"] * scale
    return restored


@torch.no_grad()
def collect_residual_components(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: Any = None,
) -> Dict[str, np.ndarray]:
    """Collect genuine residual-model components without gradient tracking."""
    model.eval()
    collected = {
        "target": [],
        "main_pred": [],
        "spectral_delta": [],
        "residual_correction": [],
        "final_pred": [],
    }
    scales = []

    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        batch_size = int(features.shape[0])
        components = model.forward_components(features)
        collected["target"].append(
            _flat_batch_tensor(targets, "target", batch_size)
        )
        for name in (
            "main_pred",
            "spectral_delta",
            "residual_correction",
            "final_pred",
        ):
            collected[name].append(
                _flat_batch_tensor(components[name], name, batch_size)
            )

        residual_scale = components["residual_scale"]
        if residual_scale.numel() != 1 or not torch.isfinite(residual_scale).all():
            raise ValueError("residual_scale must be one finite scalar.")
        scales.append(float(residual_scale.detach().cpu().reshape(-1)[0]))

    if not scales:
        raise ValueError("Test DataLoader contains no samples.")
    if not np.allclose(scales, scales[0], rtol=0.0, atol=0.0):
        raise RuntimeError("residual_scale changed during no-gradient inference.")

    flattened = {
        name: np.concatenate(batches).astype(np.float64, copy=False)
        for name, batches in collected.items()
    }
    flattened = _restore_evaluation_scale(flattened, target_scaler)
    flattened["residual_scale"] = np.asarray(scales[0], dtype=np.float64)

    expected_final = flattened["main_pred"] + flattened["residual_correction"]
    if not np.allclose(
        flattened["final_pred"],
        expected_final,
        rtol=1e-7,
        atol=1e-10,
    ):
        raise RuntimeError("Collected final predictions violate residual addition.")
    return flattened


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    """Calculate the four regression metrics required by the diagnostic."""
    calculated = calculate_all_metrics(
        target,
        prediction,
        include_directional_accuracy=False,
    )
    return {name: float(calculated[name]) for name in METRIC_NAMES}


def build_constant_control_predictions(
    test_components: Mapping[str, np.ndarray],
    validation_components: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Build label-free correction means and validation-label bias controls.

    The two correction-mean controls use model outputs only. The MSE and MAE
    biases use validation targets and validation main predictions exclusively;
    test targets are deliberately not read by this function.
    """
    test_main = np.asarray(test_components["main_pred"], dtype=np.float64)
    test_correction = np.asarray(
        test_components["residual_correction"],
        dtype=np.float64,
    )
    validation_correction = np.asarray(
        validation_components["residual_correction"],
        dtype=np.float64,
    )
    validation_target = np.asarray(
        validation_components["target"],
        dtype=np.float64,
    )
    validation_main = np.asarray(
        validation_components["main_pred"],
        dtype=np.float64,
    )

    offsets = {
        "test_correction_mean": float(np.mean(test_correction)),
        "validation_correction_mean": float(np.mean(validation_correction)),
        "validation_mse_bias": float(
            np.mean(validation_target - validation_main)
        ),
        "validation_mae_bias": float(
            np.median(validation_target - validation_main)
        ),
    }
    predictions = {
        "constant_test_correction_mean": (
            test_main + offsets["test_correction_mean"]
        ),
        "constant_validation_correction_mean": (
            test_main + offsets["validation_correction_mean"]
        ),
        "validation_mse_bias": test_main + offsets["validation_mse_bias"],
        "validation_mae_bias": test_main + offsets["validation_mae_bias"],
    }
    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError("{} contains NaN or Inf.".format(name))
    return predictions, offsets


def random_permutations(
    correction: np.ndarray,
    permutation_count: int,
    master_seed: int,
) -> Sequence[Tuple[int, int, np.ndarray]]:
    """Return reproducible random permutations with individually recorded seeds."""
    if permutation_count <= 0:
        raise ValueError("permutation_count must be positive.")
    values = np.asarray(correction, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("At least two correction values are required.")
    master_generator = np.random.default_rng(master_seed)
    seeds = master_generator.integers(
        0,
        np.iinfo(np.uint32).max,
        size=permutation_count,
        dtype=np.uint32,
    )
    return [
        (
            permutation_id,
            int(seed),
            np.random.default_rng(int(seed)).permutation(values),
        )
        for permutation_id, seed in enumerate(seeds, start=1)
    ]


def circular_shift_permutations(
    correction: np.ndarray,
) -> Sequence[Tuple[int, int, np.ndarray]]:
    """Return every nonzero circular shift, preserving correction order."""
    values = np.asarray(correction, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("At least two correction values are required.")
    return [
        (shift, shift, np.roll(values, shift))
        for shift in range(1, values.size)
    ]


def permutation_metric_rows(
    target: np.ndarray,
    main_prediction: np.ndarray,
    correction: np.ndarray,
    random_count: int = 1000,
    master_seed: int = 42,
) -> Sequence[Dict[str, Any]]:
    """Calculate metrics for random permutations and all nonzero shifts."""
    rows = []
    for permutation_id, seed, permuted in random_permutations(
        correction,
        random_count,
        master_seed,
    ):
        metrics = regression_metrics(target, main_prediction + permuted)
        rows.append(
            {
                "permutation_type": "random",
                "permutation_id": permutation_id,
                "shift": None,
                "seed": seed,
                **metrics,
            }
        )
    for permutation_id, shift, shifted in circular_shift_permutations(correction):
        metrics = regression_metrics(target, main_prediction + shifted)
        rows.append(
            {
                "permutation_type": "circular_shift",
                "permutation_id": permutation_id,
                "shift": shift,
                "seed": None,
                **metrics,
            }
        )
    return rows


def _permutation_metric_summary(
    values: np.ndarray,
    full_value: float,
    lower_is_better: bool,
) -> Dict[str, Any]:
    """Summarize one permutation metric with a direction-correct p value.

    The empirical p value estimates how often the null permutation is at least
    as good as Full, with the standard add-one correction. Thus it counts
    ``permuted <= Full`` for error metrics and ``permuted >= Full`` for R².
    Ordinary permutation p values are descriptive here because adjacent
    ``volatility_5`` targets overlap; circular shifts are a time-structure-aware
    robustness reference, not a strict significance test.
    """
    if not np.isfinite(values).all() or not np.isfinite(full_value):
        raise ValueError("Permutation metrics and Full metric must be finite.")
    if lower_is_better:
        full_better = full_value < values
        null_as_good_or_better = values <= full_value
    else:
        full_better = full_value > values
        null_as_good_or_better = values >= full_value
    return {
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "full_value": float(full_value),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
        "full_better_than_count": int(np.sum(full_better)),
        "full_better_than_proportion": float(np.mean(full_better)),
        "full_percentile_cdf": float(100.0 * np.mean(values <= full_value)),
        "empirical_p_value": float(
            (1 + np.sum(null_as_good_or_better)) / (values.size + 1)
        ),
    }


def summarize_permutations(
    rows: Sequence[Mapping[str, Any]],
    full_metrics: Mapping[str, float],
) -> Dict[str, Any]:
    """Summarize random and circular permutation metric distributions."""
    summary: Dict[str, Any] = {
        "limitations": [
            "Adjacent volatility_5 targets use overlapping future windows.",
            "Empirical permutation p values are descriptive, not strict statistical significance.",
            "Circular shifts preserve correction order but remain a robustness reference.",
        ]
    }
    for permutation_type in ("random", "circular_shift"):
        selected = [row for row in rows if row["permutation_type"] == permutation_type]
        if not selected:
            raise ValueError("Missing {} permutation rows.".format(permutation_type))
        summary[permutation_type] = {
            "count": len(selected),
            "metrics": {
                name: _permutation_metric_summary(
                    np.asarray([row[name] for row in selected], dtype=np.float64),
                    float(full_metrics[name]),
                    lower_is_better=name != "r2",
                )
                for name in METRIC_NAMES
            },
        }
    summary["full"] = {name: float(full_metrics[name]) for name in METRIC_NAMES}
    return summary


def metric_comparison(
    full_metrics: Mapping[str, float],
    control_metrics: Mapping[str, float],
) -> Dict[str, Any]:
    """Return absolute, relative, and direction-aware Full/control changes."""
    comparison = {}
    for name in METRIC_NAMES:
        full_value = float(full_metrics[name])
        control_value = float(control_metrics[name])
        denominator = abs(control_value)
        absolute_difference = full_value - control_value
        relative_difference = _safe_ratio(absolute_difference, denominator)
        improvement = (
            control_value - full_value if name != "r2" else full_value - control_value
        )
        improvement_ratio = _safe_ratio(improvement, denominator)
        comparison[name] = {
            "direction": "lower_is_better" if name != "r2" else "higher_is_better",
            "full_minus_control": absolute_difference,
            "full_minus_control_percent_of_abs_control": (
                None if relative_difference is None else 100.0 * relative_difference
            ),
            "full_improvement": improvement,
            "full_improvement_percent_of_abs_control": (
                None if improvement_ratio is None else 100.0 * improvement_ratio
            ),
        }
    return comparison


def verify_full_result(
    components: Mapping[str, np.ndarray],
    full_metrics: Mapping[str, float],
    original_metrics_path: PathLike,
    original_predictions_path: PathLike,
    tolerance: float,
) -> Dict[str, Any]:
    """Verify recomputed full predictions against the formal run artifacts."""
    metrics_path = Path(original_metrics_path)
    predictions_path = Path(original_predictions_path)
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Original metrics or prediction results are missing.")

    original_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metric_differences = {
        name: abs(float(full_metrics[name]) - float(original_metrics[name]))
        for name in METRIC_NAMES
    }
    failed_metrics = {
        name: difference
        for name, difference in metric_differences.items()
        if difference > tolerance
    }
    if failed_metrics:
        raise RuntimeError(
            "Full residual metrics do not match the formal run: {}".format(
                failed_metrics
            )
        )

    original_predictions = pd.read_csv(predictions_path)
    if len(original_predictions) != len(components["target"]):
        raise RuntimeError("Recomputed and original test sample counts differ.")
    target_difference = float(
        np.max(
            np.abs(
                original_predictions["y_true"].to_numpy(dtype=np.float64)
                - components["target"]
            )
        )
    )
    prediction_difference = float(
        np.max(
            np.abs(
                original_predictions["y_pred"].to_numpy(dtype=np.float64)
                - components["final_pred"]
            )
        )
    )
    if target_difference > tolerance or prediction_difference > tolerance:
        raise RuntimeError(
            "Recomputed predictions do not match the formal run: "
            "target max diff={}, prediction max diff={}.".format(
                target_difference,
                prediction_difference,
            )
        )

    return {
        "tolerance": tolerance,
        "metric_absolute_differences": metric_differences,
        "target_max_absolute_difference": target_difference,
        "prediction_max_absolute_difference": prediction_difference,
        "matched": True,
    }


def safe_correlation(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    """Return Pearson correlation, or ``None`` for a constant variable."""
    first_std = float(np.std(first, ddof=0))
    second_std = float(np.std(second, ddof=0))
    epsilon = np.finfo(np.float64).eps
    if first_std <= epsilon or second_std <= epsilon:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def distribution_statistics(values: np.ndarray) -> Dict[str, float]:
    """Return scale statistics for one diagnostic component."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "absolute_mean": float(np.mean(np.abs(values))),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Return a finite ratio, or ``None`` when its denominator is zero."""
    if abs(denominator) <= np.finfo(np.float64).eps:
        return None
    return float(numerator / denominator)


def contribution_summary(
    components: Mapping[str, np.ndarray],
    full_metrics: Mapping[str, float],
    main_metrics: Mapping[str, float],
    shuffled_metrics: Mapping[str, float],
    constant_metrics: Mapping[str, Mapping[str, float]],
    constant_offsets: Mapping[str, float],
    permutation_summary: Mapping[str, Any],
    source_paths: Mapping[str, Any],
) -> Dict[str, Any]:
    """Summarize residual scale, sample effects, correlations, and metric deltas."""
    target = components["target"]
    main_pred = components["main_pred"]
    spectral_delta = components["spectral_delta"]
    correction = components["residual_correction"]
    main_absolute_error = np.abs(target - main_pred)
    final_absolute_error = np.abs(target - components["final_pred"])
    improved = final_absolute_error < main_absolute_error
    worsened = final_absolute_error > main_absolute_error
    unchanged = ~(improved | worsened)
    sample_count = int(target.size)

    correction_stats = distribution_statistics(correction)
    spectral_stats = distribution_statistics(spectral_delta)
    target_std = float(np.std(target, ddof=0))
    target_absolute_mean = float(np.mean(np.abs(target)))
    main_absolute_mean = float(np.mean(np.abs(main_pred)))
    main_residual = target - main_pred
    centered_correction = correction - np.mean(correction)

    return {
        "source": dict(source_paths),
        "interpretation_boundaries": {
            "main_only_definition": (
                "main-only is the main branch of the jointly trained residual model"
            ),
            "not_plain_lstm": (
                "main-only is not an independently trained Plain LSTM baseline"
            ),
            "allowed_inference": (
                "Full versus main-only measures whether correction changes the jointly trained model"
            ),
            "disallowed_inference": (
                "This diagnostic cannot establish that the residual model "
                "outperforms an independently trained Plain LSTM"
            ),
            "permutation_limit": (
                "One shuffled correction is descriptive only; repeated random and circular distributions are required"
            ),
        },
        "sample_count": sample_count,
        "residual_scale": float(components["residual_scale"]),
        "spectral_delta": spectral_stats,
        "residual_correction": correction_stats,
        "demeaned_residual_correction": {
            **distribution_statistics(centered_correction),
            "correlation_with_target_minus_main_pred": safe_correlation(
                centered_correction,
                main_residual,
            ),
        },
        "scale_ratios": {
            "correction_std_over_target_std": _safe_ratio(
                correction_stats["std"], target_std
            ),
            "correction_absolute_mean_over_target_absolute_mean": _safe_ratio(
                correction_stats["absolute_mean"], target_absolute_mean
            ),
            "correction_absolute_mean_over_main_prediction_absolute_mean": _safe_ratio(
                correction_stats["absolute_mean"], main_absolute_mean
            ),
        },
        "nonzero_correction": {
            "count": int(np.count_nonzero(correction)),
            "proportion": float(np.count_nonzero(correction) / sample_count),
        },
        "sample_absolute_error_effect": {
            "improved_count": int(np.sum(improved)),
            "improved_proportion": float(np.mean(improved)),
            "worsened_count": int(np.sum(worsened)),
            "worsened_proportion": float(np.mean(worsened)),
            "unchanged_count": int(np.sum(unchanged)),
            "unchanged_proportion": float(np.mean(unchanged)),
        },
        "correlations": {
            "residual_correction_with_target_minus_main_pred": safe_correlation(
                correction, main_residual
            ),
            "spectral_delta_with_target_minus_main_pred": safe_correlation(
                spectral_delta, main_residual
            ),
        },
        "full_minus_main_only": {
            name: float(full_metrics[name] - main_metrics[name])
            for name in ("mae", "rmse", "r2")
        },
        "full_minus_shuffled_correction": {
            name: float(full_metrics[name] - shuffled_metrics[name])
            for name in ("mae", "rmse", "r2")
        },
        "constant_offsets": {
            name: float(value) for name, value in constant_offsets.items()
        },
        "full_vs_constant_controls": {
            name: metric_comparison(full_metrics, metrics)
            for name, metrics in constant_metrics.items()
        },
        "permutation_position": {
            permutation_type: {
                metric_name: {
                    key: metric_summary[key]
                    for key in (
                        "direction",
                        "full_percentile_cdf",
                        "full_better_than_proportion",
                        "empirical_p_value",
                    )
                }
                for metric_name, metric_summary in permutation_summary[
                    permutation_type
                ]["metrics"].items()
            }
            for permutation_type in ("random", "circular_shift")
        },
    }


def regime_rows(
    components: Mapping[str, np.ndarray],
) -> Tuple[Sequence[Dict[str, Any]], Dict[str, float]]:
    """Compare main and full errors across target-volatility quartile regimes."""
    target = components["target"]
    main_pred = components["main_pred"]
    final_pred = components["final_pred"]
    correction = components["residual_correction"]
    centered_correction = correction - np.mean(correction)
    main_absolute_error = np.abs(target - main_pred)
    final_absolute_error = np.abs(target - final_pred)
    improved = final_absolute_error < main_absolute_error
    q25, q75 = np.quantile(target, [0.25, 0.75])
    low_mask = target <= q25
    high_mask = (target >= q75) & ~low_mask
    normal_mask = ~(low_mask | high_mask)
    masks = (
        ("Low volatility", low_mask),
        ("Normal volatility", normal_mask),
        ("High volatility", high_mask),
    )

    rows = []
    for name, mask in masks:
        main = regression_metrics(target[mask], main_pred[mask])
        full = regression_metrics(target[mask], final_pred[mask])
        rows.append(
            {
                "regime": name,
                "sample_count": int(np.sum(mask)),
                "main_mae": main["mae"],
                "full_mae": full["mae"],
                "main_rmse": main["rmse"],
                "full_rmse": full["rmse"],
                "correction_improved_count": int(np.sum(improved[mask])),
                "correction_improved_proportion": float(np.mean(improved[mask])),
                "residual_correction_absolute_mean": float(
                    np.mean(np.abs(correction[mask]))
                ),
                "demeaned_correction_mean": float(
                    np.mean(centered_correction[mask])
                ),
                "demeaned_correction_std": float(
                    np.std(centered_correction[mask], ddof=0)
                ),
                "demeaned_correction_absolute_mean": float(
                    np.mean(np.abs(centered_correction[mask]))
                ),
            }
        )
    return rows, {"target_q25": float(q25), "target_q75": float(q75)}


def create_unique_output_dir(output_root: PathLike, run_id: str) -> Path:
    """Create a non-overwriting diagnostic directory for one checkpoint run."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / run_id
    if candidate.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = root / "{}_{}".format(run_id, suffix)
        counter = 1
        while candidate.exists():
            candidate = root / "{}_{}_{:02d}".format(run_id, suffix, counter)
            counter += 1
    candidate.mkdir(parents=False, exist_ok=False)
    (candidate / "figures").mkdir(parents=False, exist_ok=False)
    return candidate


def save_components_csv(
    components: Mapping[str, np.ndarray],
    target_dates: Optional[Sequence[Any]],
    path: PathLike,
) -> None:
    """Save per-sample prediction components and error effects."""
    target = components["target"]
    main_pred = components["main_pred"]
    final_pred = components["final_pred"]
    main_error = target - main_pred
    final_error = target - final_pred
    frame_data: Dict[str, Any] = {"sample_index": np.arange(target.size)}
    if target_dates is not None:
        if len(target_dates) != target.size:
            raise ValueError("target_dates length does not match test samples.")
        frame_data["target_date"] = [
            pd.Timestamp(date).strftime("%Y-%m-%d") for date in target_dates
        ]
    frame_data.update(
        {
            "target": target,
            "main_pred": main_pred,
            "spectral_delta": components["spectral_delta"],
            "residual_scale": np.full(
                target.size,
                float(components["residual_scale"]),
            ),
            "residual_correction": components["residual_correction"],
            "final_pred": final_pred,
            "main_error": main_error,
            "final_error": final_error,
            "main_absolute_error": np.abs(main_error),
            "final_absolute_error": np.abs(final_error),
            "correction_improved": np.abs(final_error) < np.abs(main_error),
        }
    )
    pd.DataFrame(frame_data).to_csv(path, index=False, encoding="utf-8-sig")


def save_regime_csv(rows: Sequence[Mapping[str, Any]], path: PathLike) -> None:
    """Save regime-level comparison rows as UTF-8 CSV."""
    if not rows:
        raise ValueError("Regime rows cannot be empty.")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_permutation_csv(
    rows: Sequence[Mapping[str, Any]],
    path: PathLike,
) -> None:
    """Save every random-permutation and circular-shift metric row."""
    if not rows:
        raise ValueError("Permutation rows cannot be empty.")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: Mapping[str, Any], path: PathLike) -> None:
    """Save diagnostic data as indented UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def _save_figure(figure: plt.Figure, figures_dir: Path, stem: str) -> None:
    """Save one unaltered diagnostic figure as PNG and PDF."""
    figure.tight_layout()
    figure.savefig(
        figures_dir / "{}.png".format(stem),
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        figures_dir / "{}.pdf".format(stem),
        bbox_inches="tight",
    )
    plt.close(figure)


def create_diagnostic_figures(
    components: Mapping[str, np.ndarray],
    regimes: Sequence[Mapping[str, Any]],
    figures_dir: PathLike,
) -> None:
    """Create the five required component and regime diagnostic figures."""
    output_dir = Path(figures_dir)
    index = np.arange(components["target"].size)
    target = components["target"]
    main_pred = components["main_pred"]
    final_pred = components["final_pred"]
    correction = components["residual_correction"]
    main_residual = target - main_pred

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(index, target, label="Target", linewidth=1.0)
    axis.plot(index, main_pred, label="Main prediction", linewidth=0.9)
    axis.plot(index, final_pred, label="Final prediction", linewidth=0.9)
    axis.set(xlabel="Test sample", ylabel="Volatility", title="Main vs Final Prediction")
    axis.legend()
    axis.grid(alpha=0.25)
    _save_figure(figure, output_dir, "main_vs_final_prediction")

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(index, correction, color="tab:orange", linewidth=0.9)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set(ylabel="Correction", title="Residual Correction Series")
    axes[0].grid(alpha=0.25)
    axes[1].plot(index, target, color="tab:blue", linewidth=0.9)
    axes[1].set(xlabel="Test sample", ylabel="Target volatility")
    axes[1].grid(alpha=0.25)
    _save_figure(figure, output_dir, "residual_correction_series")

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(index, np.abs(main_residual), label="Main absolute error", linewidth=0.9)
    axis.plot(
        index,
        np.abs(target - final_pred),
        label="Final absolute error",
        linewidth=0.9,
    )
    axis.set(xlabel="Test sample", ylabel="Absolute error", title="Main vs Final Absolute Error")
    axis.legend()
    axis.grid(alpha=0.25)
    _save_figure(figure, output_dir, "main_vs_final_absolute_error")

    correlation = safe_correlation(correction, main_residual)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(main_residual, correction, s=12, alpha=0.6)
    label = "undefined" if correlation is None else "{:.6f}".format(correlation)
    axis.set(
        xlabel="Target - main prediction",
        ylabel="Residual correction",
        title="Correction vs Main Residual (r={})".format(label),
    )
    axis.grid(alpha=0.25)
    _save_figure(figure, output_dir, "correction_vs_main_residual")

    regime_names = [str(row["regime"]) for row in regimes]
    positions = np.arange(len(regime_names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        positions - width / 2,
        [float(row["main_mae"]) for row in regimes],
        width,
        label="Main MAE",
    )
    axis.bar(
        positions + width / 2,
        [float(row["full_mae"]) for row in regimes],
        width,
        label="Full MAE",
    )
    axis.set_xticks(positions, regime_names)
    axis.set(ylabel="MAE", title="Regime Error Comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir, "regime_error_comparison")


def create_extended_diagnostic_figures(
    permutation_rows: Sequence[Mapping[str, Any]],
    full_metrics: Mapping[str, float],
    comparison_metrics: Mapping[str, Mapping[str, float]],
    figures_dir: PathLike,
) -> None:
    """Plot permutation distributions and constant-bias comparisons."""
    output_dir = Path(figures_dir)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, metric_name in zip(axes.reshape(-1), METRIC_NAMES):
        random_values = np.asarray(
            [
                row[metric_name]
                for row in permutation_rows
                if row["permutation_type"] == "random"
            ],
            dtype=np.float64,
        )
        circular_values = np.asarray(
            [
                row[metric_name]
                for row in permutation_rows
                if row["permutation_type"] == "circular_shift"
            ],
            dtype=np.float64,
        )
        axis.hist(random_values, bins=35, alpha=0.55, label="Random")
        axis.hist(circular_values, bins=35, alpha=0.55, label="Circular shift")
        axis.axvline(
            float(full_metrics[metric_name]),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Full",
        )
        direction = "lower is better" if metric_name != "r2" else "higher is better"
        axis.set(
            xlabel=metric_name.upper(),
            ylabel="Count",
            title="{} ({})".format(metric_name.upper(), direction),
        )
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Permutation Metric Distributions with Full Reference", y=1.01)
    _save_figure(figure, output_dir, "permutation_metric_distribution")

    labels = list(comparison_metrics)
    display_labels = [label.replace("_", " ") for label in labels]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    for axis, metric_name in zip(axes.reshape(-1), METRIC_NAMES):
        values = [float(comparison_metrics[label][metric_name]) for label in labels]
        axis.bar(positions, values, color="tab:blue", alpha=0.8)
        axis.axhline(
            float(full_metrics[metric_name]),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Full reference",
        )
        axis.set_xticks(positions, display_labels, rotation=25, ha="right")
        direction = "lower is better" if metric_name != "r2" else "higher is better"
        axis.set(
            ylabel=metric_name.upper(),
            title="{} ({})".format(metric_name.upper(), direction),
        )
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    figure.suptitle("Constant and Validation Bias Controls", y=1.01)
    _save_figure(figure, output_dir, "constant_bias_comparison")


def run_diagnostic(
    config_path: PathLike = DEFAULT_CONFIG,
    checkpoint_path: PathLike = DEFAULT_CHECKPOINT,
    original_metrics_path: PathLike = DEFAULT_ORIGINAL_METRICS,
    original_predictions_path: PathLike = DEFAULT_ORIGINAL_PREDICTIONS,
    output_root: PathLike = DEFAULT_OUTPUT_ROOT,
    seed: int = 42,
    num_permutations: int = 1000,
    tolerance: float = 1e-8,
    output_dir: Optional[PathLike] = None,
) -> Path:
    """Run all no-training residual contribution diagnostics."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    config = load_config(config_path)
    data_config = config["data"]
    training_config = config["training"]
    model_config = config["model"]
    model_type = str(model_config.get("type", "")).strip().lower()
    supported_model_types = {
        "residual_auxiliary_lct_riesz_lstm",
        "residual_auxiliary_signal_lstm",
    }
    if model_type not in supported_model_types:
        raise ValueError(
            "Config must describe a supported residual auxiliary model."
        )

    seed = int(training_config["seed"])
    set_seed(seed)
    device = resolve_device(training_config.get("device", "auto"))
    data_bundle: FinancialDataLoaders = create_dataloaders(
        csv_path=str(data_config["csv_path"]),
        sequence_length=int(data_config["sequence_length"]),
        batch_size=int(data_config["batch_size"]),
        feature_columns=data_config.get("feature_columns"),
        derived_features=data_config.get("derived_features"),
        target_type=str(data_config["target_type"]),
        train_ratio=float(data_config["train_ratio"]),
        val_ratio=float(data_config["val_ratio"]),
        test_ratio=float(data_config["test_ratio"]),
        shuffle_train=False,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_config).to(device)
    if not callable(getattr(model, "forward_components", None)):
        raise TypeError("Configured model does not expose residual components.")
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    components = collect_residual_components(
        model,
        data_bundle.test_loader,
        device,
        target_scaler=data_bundle.target_scaler,
    )
    validation_components = collect_residual_components(
        model,
        data_bundle.val_loader,
        device,
        target_scaler=data_bundle.target_scaler,
    )
    full_metrics = regression_metrics(components["target"], components["final_pred"])
    verification = verify_full_result(
        components,
        full_metrics,
        original_metrics_path,
        original_predictions_path,
        tolerance,
    )

    main_metrics = regression_metrics(components["target"], components["main_pred"])
    generator = np.random.default_rng(seed)
    shuffled_correction = generator.permutation(components["residual_correction"])
    shuffled_prediction = components["main_pred"] + shuffled_correction
    shuffled_metrics = regression_metrics(components["target"], shuffled_prediction)
    constant_predictions, constant_offsets = build_constant_control_predictions(
        components,
        validation_components,
    )
    constant_metrics = {
        name: regression_metrics(components["target"], prediction)
        for name, prediction in constant_predictions.items()
    }
    permutation_rows = permutation_metric_rows(
        components["target"],
        components["main_pred"],
        components["residual_correction"],
        random_count=num_permutations,
        master_seed=seed,
    )
    permutation_summary = summarize_permutations(
        permutation_rows,
        full_metrics,
    )
    interpretation_boundaries = {
        "main_only": (
            "The jointly trained residual model's main branch, not an independently trained Plain LSTM."
        ),
        "full_vs_main_only": (
            "Shows whether correction affects the jointly trained model only."
        ),
        "plain_lstm_claim": (
            "No Full/main-only comparison can establish that this residual "
            "model outperforms an independently trained Plain LSTM."
        ),
        "permutation_claim": (
            "A single shuffle is descriptive; repeated random and circular distributions define robustness."
        ),
    }
    metrics_document = {
        "model": {
            "type": model_type,
            "use_lct_riesz": bool(getattr(model, "use_lct_riesz", False)),
        },
        "interpretation_boundaries": interpretation_boundaries,
        "verification": verification,
        "full": full_metrics,
        "main_only_jointly_trained_branch": main_metrics,
        "main_only": main_metrics,
        "constant_test_correction_mean": constant_metrics[
            "constant_test_correction_mean"
        ],
        "constant_validation_correction_mean": constant_metrics[
            "constant_validation_correction_mean"
        ],
        "validation_mse_bias": constant_metrics["validation_mse_bias"],
        "validation_mae_bias": constant_metrics["validation_mae_bias"],
        "single_shuffled_correction": shuffled_metrics,
        "shuffled_correction": shuffled_metrics,
        "constant_offsets": constant_offsets,
        "full_vs_constant_controls": {
            name: metric_comparison(full_metrics, metrics)
            for name, metrics in constant_metrics.items()
        },
        "permutation_summary": permutation_summary,
        "shuffle_master_seed": int(seed),
        "random_permutation_count": int(num_permutations),
        "circular_shift_count": int(len(components["target"]) - 1),
    }

    source_paths = {
        "config": str(Path(config_path)),
        "checkpoint": str(Path(checkpoint_path)),
        "original_metrics": str(Path(original_metrics_path)),
        "original_predictions": str(Path(original_predictions_path)),
        "model_type": model_type,
        "use_lct_riesz": bool(getattr(model, "use_lct_riesz", False)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
    }
    summary = contribution_summary(
        components,
        full_metrics,
        main_metrics,
        shuffled_metrics,
        constant_metrics,
        constant_offsets,
        permutation_summary,
        source_paths,
    )
    regimes, thresholds = regime_rows(components)
    summary["regime_thresholds"] = thresholds

    run_id = Path(checkpoint_path).parent.name
    protected_directories = {
        Path(original_metrics_path).resolve().parent,
        Path(original_predictions_path).resolve().parent,
        Path(checkpoint_path).resolve().parent,
    }
    requested_output_root = Path(
        output_root if output_dir is None else output_dir
    ).resolve()
    if any(
        requested_output_root == protected
        or protected in requested_output_root.parents
        for protected in protected_directories
    ):
        raise ValueError(
            "Diagnostic output must not be inside an original experiment "
            "or checkpoint directory."
        )
    if output_dir is None:
        resolved_output_dir = create_unique_output_dir(output_root, run_id)
    else:
        resolved_output_dir = Path(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        (resolved_output_dir / "figures").mkdir(parents=True, exist_ok=True)
    save_components_csv(
        components,
        data_bundle.test_dataset.target_dates,
        resolved_output_dir / "residual_components.csv",
    )
    save_json(metrics_document, resolved_output_dir / "residual_ablation_metrics.json")
    save_json(summary, resolved_output_dir / "residual_contribution_summary.json")
    save_regime_csv(regimes, resolved_output_dir / "residual_regime_metrics.csv")
    save_permutation_csv(
        permutation_rows,
        resolved_output_dir / "permutation_metrics.csv",
    )
    save_json(permutation_summary, resolved_output_dir / "permutation_summary.json")
    create_diagnostic_figures(
        components,
        regimes,
        resolved_output_dir / "figures",
    )
    comparison_metrics = {
        "full": full_metrics,
        "main_only_jointly_trained": main_metrics,
        **constant_metrics,
    }
    create_extended_diagnostic_figures(
        permutation_rows,
        full_metrics,
        comparison_metrics,
        resolved_output_dir / "figures",
    )
    return resolved_output_dir


def main() -> None:
    """Parse CLI arguments and execute the no-training diagnostic."""
    parser = argparse.ArgumentParser(
        description="Diagnose residual LCT-Riesz contribution from a checkpoint.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Saved experiment config JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Trained residual checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--original-metrics",
        default=str(DEFAULT_ORIGINAL_METRICS),
        help="Formal metrics.json used for exact-result verification.",
    )
    parser.add_argument(
        "--original-predictions",
        default=str(DEFAULT_ORIGINAL_PREDICTIONS),
        help="Formal prediction_results.csv used for exact verification.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root for a non-overwriting diagnostic directory.",
    )
    parser.add_argument(
        "--output-dir",
        help="Exact diagnostic output directory; overrides --output-root.",
    )
    parser.add_argument(
        "--num-permutations",
        "--random-permutations",
        dest="num_permutations",
        type=int,
        default=1000,
        help="Number of random correction permutations (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        "--shuffle-seed",
        dest="seed",
        type=int,
        default=42,
        help="Master seed for all diagnostic randomization (default: %(default)s).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Maximum allowed Full/formal absolute difference.",
    )
    arguments = parser.parse_args()
    output_dir = run_diagnostic(
        config_path=arguments.config,
        checkpoint_path=arguments.checkpoint,
        original_metrics_path=arguments.original_metrics,
        original_predictions_path=arguments.original_predictions,
        output_root=arguments.output_root,
        seed=arguments.seed,
        num_permutations=arguments.num_permutations,
        tolerance=arguments.tolerance,
        output_dir=arguments.output_dir,
    )
    print("Diagnostic artifacts saved to {}".format(output_dir))


if __name__ == "__main__":
    main()
