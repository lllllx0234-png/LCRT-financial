"""Regression metrics for financial time-series forecasting."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch


ArrayLike = Union[Sequence[float], np.ndarray, torch.Tensor]


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean absolute error."""
    true_values, predicted_values = _prepare_pair(y_true, y_pred)
    return float(np.mean(np.abs(true_values - predicted_values)))


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean squared error."""
    true_values, predicted_values = _prepare_pair(y_true, y_pred)
    return float(np.mean(np.square(true_values - predicted_values)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    epsilon: float = 1e-8,
) -> float:
    """Return mean absolute percentage error with a stable denominator."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    true_values, predicted_values = _prepare_pair(y_true, y_pred)
    denominator = np.maximum(np.abs(true_values), epsilon)
    percentage_errors = np.abs(true_values - predicted_values) / denominator
    return float(np.mean(percentage_errors) * 100.0)


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return R-squared, with finite behavior for constant targets."""
    true_values, predicted_values = _prepare_pair(y_true, y_pred)
    residual_sum = float(np.sum(np.square(true_values - predicted_values)))
    total_sum = float(
        np.sum(np.square(true_values - np.mean(true_values)))
    )

    if np.isclose(total_sum, 0.0):
        return 1.0 if np.isclose(residual_sum, 0.0) else 0.0
    return float(1.0 - residual_sum / total_sum)


def directional_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the proportion of matching true and predicted return signs."""
    true_values, predicted_values = _prepare_pair(y_true, y_pred)
    matching_directions = np.sign(true_values) == np.sign(predicted_values)
    return float(np.mean(matching_directions))


def calculate_all_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Dict[str, float]:
    """Calculate all standard regression and directional metrics."""
    true_values, predicted_values = _prepare_pair(y_true, y_pred)
    return {
        "mae": mae(true_values, predicted_values),
        "mse": mse(true_values, predicted_values),
        "rmse": rmse(true_values, predicted_values),
        "mape": mape(true_values, predicted_values),
        "r2": r2_score(true_values, predicted_values),
        "directional_accuracy": directional_accuracy(
            true_values,
            predicted_values,
        ),
    }


def _prepare_pair(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert two array-like inputs to aligned finite one-dimensional arrays."""
    true_values = _to_flat_numpy(y_true, "y_true")
    predicted_values = _to_flat_numpy(y_pred, "y_pred")
    if true_values.shape != predicted_values.shape:
        raise ValueError("y_true and y_pred must have the same number of values.")
    if true_values.size == 0:
        raise ValueError("y_true and y_pred cannot be empty.")
    return true_values, predicted_values


def _to_flat_numpy(values: ArrayLike, name: str) -> np.ndarray:
    """Convert a NumPy, PyTorch, or sequence input to finite float64 values."""
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    flattened = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.isfinite(flattened).all():
        raise ValueError("{} contains non-finite values.".format(name))
    return flattened


__all__ = [
    "calculate_all_metrics",
    "directional_accuracy",
    "mae",
    "mape",
    "mse",
    "r2_score",
    "rmse",
]
