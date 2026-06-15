"""Publication-oriented visualizations for forecasting experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


ArrayLike = Union[Sequence[float], np.ndarray, torch.Tensor]
PathLike = Union[str, Path]

_FIGURE_SIZE = (8.0, 5.0)
_DPI = 300


def plot_loss_curves(
    train_losses: ArrayLike,
    val_losses: ArrayLike,
    save_path: PathLike,
) -> Tuple[Path, Path]:
    """Plot training and validation loss by epoch and save PNG/PDF files."""
    train_values, val_values = _prepare_pair(train_losses, val_losses)
    epochs = np.arange(1, train_values.size + 1)

    figure, axis = plt.subplots(figsize=_FIGURE_SIZE)
    axis.plot(epochs, train_values, label="Train Loss", linewidth=1.8)
    axis.plot(epochs, val_values, label="Validation Loss", linewidth=1.8)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training and Validation Loss")
    axis.legend()
    axis.grid(True, alpha=0.3)
    return _save_figure(figure, save_path, "loss_curve")


def plot_prediction_curve(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    save_path: PathLike,
) -> Tuple[Path, Path]:
    """Plot ground-truth and predicted values across the time index."""
    true_values, predicted_values = _prepare_pair(y_true, y_pred)
    time_index = np.arange(true_values.size)

    figure, axis = plt.subplots(figsize=_FIGURE_SIZE)
    axis.plot(
        time_index,
        true_values,
        label="Ground Truth",
        linewidth=1.8,
    )
    axis.plot(
        time_index,
        predicted_values,
        label="Prediction",
        linewidth=1.6,
        alpha=0.85,
    )
    axis.set_xlabel("Time Index")
    axis.set_ylabel("Value")
    axis.set_title("Prediction vs Ground Truth")
    axis.legend()
    axis.grid(True, alpha=0.3)
    return _save_figure(figure, save_path, "prediction_curve")


def plot_residual_distribution(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    save_path: PathLike,
) -> Tuple[Path, Path]:
    """Plot residuals across time using residual = y_true - y_pred."""
    true_values, predicted_values = _prepare_pair(y_true, y_pred)
    residuals = true_values - predicted_values
    time_index = np.arange(residuals.size)

    figure, axis = plt.subplots(figsize=_FIGURE_SIZE)
    axis.plot(time_index, residuals, color="#4C72B0", linewidth=1.2)
    axis.scatter(time_index, residuals, color="#4C72B0", s=12, alpha=0.65)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Time Index")
    axis.set_ylabel("Residual")
    axis.set_title("Residual Distribution")
    axis.grid(True, alpha=0.3)
    return _save_figure(figure, save_path, "residual_distribution")


def plot_error_histogram(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    save_path: PathLike,
) -> Tuple[Path, Path]:
    """Plot a histogram using prediction error = y_pred - y_true."""
    true_values, predicted_values = _prepare_pair(y_true, y_pred)
    errors = predicted_values - true_values
    bin_count = min(30, max(5, int(np.sqrt(errors.size))))

    figure, axis = plt.subplots(figsize=_FIGURE_SIZE)
    axis.hist(
        errors,
        bins=bin_count,
        color="#55A868",
        edgecolor="black",
        alpha=0.8,
    )
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Prediction Error")
    axis.set_ylabel("Frequency")
    axis.set_title("Prediction Error Histogram")
    axis.grid(True, axis="y", alpha=0.3)
    return _save_figure(figure, save_path, "error_histogram")


def plot_lct_parameter_history(
    parameter_history: Sequence[Mapping[str, float]],
    save_path: PathLike,
) -> Tuple[Path, Path]:
    """Plot alpha, m, q, and gamma values across training epochs."""
    if not parameter_history:
        raise ValueError("parameter_history cannot be empty.")

    required_fields = ("epoch", "alpha", "m", "q", "gamma")
    parsed: Dict[str, List[float]] = {
        field: [] for field in required_fields
    }
    for row_index, row in enumerate(parameter_history):
        missing_fields = [field for field in required_fields if field not in row]
        if missing_fields:
            raise ValueError(
                "parameter_history row {} is missing fields: {}".format(
                    row_index,
                    missing_fields,
                )
            )
        for field in required_fields:
            value = float(row[field])
            if not np.isfinite(value):
                raise ValueError(
                    "{} contains a non-finite value.".format(field)
                )
            parsed[field].append(value)

    figure, axis = plt.subplots(figsize=_FIGURE_SIZE)
    for field in ("alpha", "m", "q", "gamma"):
        axis.plot(
            parsed["epoch"],
            parsed[field],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=field,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Parameter Value")
    axis.set_title("Learned LCT-Riesz Parameter History")
    axis.legend(ncol=2)
    axis.grid(True, alpha=0.3)
    return _save_figure(figure, save_path, "lct_parameter_analysis")


def _prepare_pair(
    first: ArrayLike,
    second: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert and validate two equally sized finite plotting arrays."""
    first_values = _to_flat_numpy(first, "first input")
    second_values = _to_flat_numpy(second, "second input")
    if first_values.shape != second_values.shape:
        raise ValueError("Plotting inputs must have equal lengths.")
    if first_values.size == 0:
        raise ValueError("Plotting inputs cannot be empty.")
    return first_values, second_values


def _to_flat_numpy(values: ArrayLike, name: str) -> np.ndarray:
    """Convert a tensor or array-like input into finite float64 values."""
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    flattened = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.isfinite(flattened).all():
        raise ValueError("{} contains non-finite values.".format(name))
    return flattened


def _save_figure(
    figure: plt.Figure,
    save_path: PathLike,
    filename_stem: str,
) -> Tuple[Path, Path]:
    """Save one figure as 300 DPI PNG and vector PDF, then close it."""
    output_directory = Path(save_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / "{}.png".format(filename_stem)
    pdf_path = output_directory / "{}.pdf".format(filename_stem)

    figure.tight_layout()
    try:
        figure.savefig(
            png_path,
            dpi=_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        figure.savefig(
            pdf_path,
            dpi=_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
    finally:
        plt.close(figure)
    return png_path, pdf_path


__all__ = [
    "plot_error_histogram",
    "plot_lct_parameter_history",
    "plot_loss_curves",
    "plot_prediction_curve",
    "plot_residual_distribution",
]
