"""Reproducible experiment directory and artifact persistence utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ExperimentPaths:
    """Paths belonging to one isolated experiment run."""

    experiment_dir: Path
    checkpoint_dir: Path
    figures_dir: Path
    config_path: Path
    training_log_path: Path
    metrics_path: Path
    prediction_results_path: Path
    summary_path: Path
    lct_parameters_path: Path
    best_model_path: Path
    latest_model_path: Path


def create_experiment_dir(
    outputs_root: PathLike = "outputs",
    checkpoints_root: PathLike = "checkpoints",
    timestamp: Optional[datetime] = None,
    prefix: str = "experiment",
) -> ExperimentPaths:
    """Create unique output, checkpoint, and figure directories for one run."""
    normalized_prefix = prefix.strip()
    if not normalized_prefix:
        raise ValueError("prefix cannot be empty.")
    if any(character in normalized_prefix for character in ("\\", "/", ":")):
        raise ValueError("prefix cannot contain path separators or a drive marker.")

    outputs_directory = Path(outputs_root)
    checkpoints_directory = Path(checkpoints_root)
    outputs_directory.mkdir(parents=True, exist_ok=True)
    checkpoints_directory.mkdir(parents=True, exist_ok=True)

    run_time = timestamp if timestamp is not None else datetime.now()
    base_name = "{}_{}".format(
        normalized_prefix,
        run_time.strftime("%Y%m%d_%H%M%S"),
    )
    experiment_name = _find_unique_experiment_name(
        base_name,
        outputs_directory,
        checkpoints_directory,
    )

    experiment_dir = outputs_directory / experiment_name
    checkpoint_dir = checkpoints_directory / experiment_name
    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    return ExperimentPaths(
        experiment_dir=experiment_dir,
        checkpoint_dir=checkpoint_dir,
        figures_dir=figures_dir,
        config_path=experiment_dir / "config.json",
        training_log_path=experiment_dir / "training_log.csv",
        metrics_path=experiment_dir / "metrics.json",
        prediction_results_path=experiment_dir / "prediction_results.csv",
        summary_path=experiment_dir / "experiment_summary.txt",
        lct_parameters_path=experiment_dir / "learned_lct_parameters.txt",
        best_model_path=checkpoint_dir / "best_model.pth",
        latest_model_path=checkpoint_dir / "latest_model.pth",
    )


def save_config(config: Mapping[str, Any], path: PathLike) -> None:
    """Save an experiment configuration as indented UTF-8 JSON."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")
    _save_json(config, path)


def append_training_log(log_row: Mapping[str, Any], path: PathLike) -> None:
    """Append one epoch log row, creating or extending the CSV header."""
    if not isinstance(log_row, Mapping) or not log_row:
        raise ValueError("log_row must be a non-empty mapping.")

    required_fields = {"epoch", "train_loss", "val_loss", "learning_rate"}
    missing_fields = required_fields.difference(log_row)
    if missing_fields:
        raise ValueError(
            "log_row is missing required fields: {}".format(
                sorted(missing_fields)
            )
        )

    output_path = _prepare_parent(path)
    serializable_row = {
        str(key): _to_serializable(value) for key, value in log_row.items()
    }

    if not output_path.exists():
        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(serializable_row))
            writer.writeheader()
            writer.writerow(serializable_row)
        return

    with output_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        existing_fields = list(reader.fieldnames or [])
        existing_rows = list(reader)

    combined_fields = existing_fields + [
        field for field in serializable_row if field not in existing_fields
    ]
    if combined_fields != existing_fields:
        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=combined_fields)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(serializable_row)
    else:
        with output_path.open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=combined_fields)
            writer.writerow(serializable_row)


def save_metrics(metrics: Mapping[str, Any], path: PathLike) -> None:
    """Save evaluation metrics as indented UTF-8 JSON."""
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping.")
    _save_json(metrics, path)


def save_prediction_results(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    path: PathLike,
) -> None:
    """Save true values, predictions, and residual errors as CSV."""
    true_values = _as_flat_numpy(y_true, "y_true")
    predicted_values = _as_flat_numpy(y_pred, "y_pred")
    if true_values.shape != predicted_values.shape:
        raise ValueError("y_true and y_pred must have the same number of values.")

    output_path = _prepare_parent(path)
    errors = predicted_values - true_values
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["y_true", "y_pred", "error"])
        writer.writerows(
            zip(
                true_values.tolist(),
                predicted_values.tolist(),
                errors.tolist(),
            )
        )


def save_lct_parameters(params: Mapping[str, Any], path: PathLike) -> None:
    """Save learned LCT-Riesz parameters in a thesis-readable text format."""
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping.")
    required_fields = ("alpha", "m", "q", "A", "B", "C", "D", "gamma")
    missing_fields = [field for field in required_fields if field not in params]
    if missing_fields:
        raise ValueError(
            "LCT parameters are missing required fields: {}".format(
                missing_fields
            )
        )

    values = {
        field: float(_to_serializable(params[field])) for field in required_fields
    }
    content = (
        "Learned One-Dimensional LCT-Riesz Parameters\n"
        "============================================\n\n"
        "Temporal LCT parameterization\n"
        "-----------------------------\n"
        "alpha = {alpha:.10g}\n"
        "m     = {m:.10g}\n"
        "q     = {q:.10g}\n"
        "gamma = {gamma:.10g}\n\n"
        "LCT matrix\n"
        "----------\n"
        "[ A  B ] = [ {A:.10g}  {B:.10g} ]\n"
        "[ C  D ]   [ {C:.10g}  {D:.10g} ]\n\n"
        "determinant (A*D - B*C) = {determinant:.10g}\n"
    ).format(
        determinant=values["A"] * values["D"] - values["B"] * values["C"],
        **values
    )
    _save_text(content, path)


def save_experiment_summary(
    summary: Union[str, Mapping[str, Any]],
    path: PathLike,
) -> None:
    """Save a free-form or structured experiment summary as UTF-8 text."""
    if isinstance(summary, str):
        content = summary
    elif isinstance(summary, Mapping):
        lines = ["Experiment Summary", "==================", ""]
        lines.extend(
            "{}: {}".format(key, _format_summary_value(value))
            for key, value in summary.items()
        )
        content = "\n".join(lines)
    else:
        raise TypeError("summary must be a string or mapping.")

    if not content.endswith("\n"):
        content += "\n"
    _save_text(content, path)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, Any],
    path: PathLike,
) -> None:
    """Save model, optimizer, epoch, and metrics in a PyTorch checkpoint."""
    if epoch < 0:
        raise ValueError("epoch cannot be negative.")
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping.")

    output_path = _prepare_parent(path)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": _to_serializable(metrics),
    }
    torch.save(checkpoint, output_path)


def _find_unique_experiment_name(
    base_name: str,
    outputs_root: Path,
    checkpoints_root: Path,
) -> str:
    """Find a run name unused by both output and checkpoint roots."""
    candidate = base_name
    suffix = 1
    while (outputs_root / candidate).exists() or (
        checkpoints_root / candidate
    ).exists():
        candidate = "{}_{:02d}".format(base_name, suffix)
        suffix += 1
    return candidate


def _prepare_parent(path: PathLike) -> Path:
    """Create a file's parent directory and return its Path object."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _save_json(data: Mapping[str, Any], path: PathLike) -> None:
    """Serialize a mapping to readable UTF-8 JSON."""
    output_path = _prepare_parent(path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            _to_serializable(data),
            file,
            ensure_ascii=False,
            indent=4,
        )
        file.write("\n")


def _save_text(content: str, path: PathLike) -> None:
    """Save UTF-8 text after ensuring its parent directory exists."""
    output_path = _prepare_parent(path)
    output_path.write_text(content, encoding="utf-8")


def _to_serializable(value: Any) -> Any:
    """Convert common scientific Python values into JSON-compatible objects."""
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        "Value of type {} is not JSON serializable.".format(type(value).__name__)
    )


def _as_flat_numpy(values: Sequence[Any], name: str) -> np.ndarray:
    """Convert an array-like object or tensor to a finite one-dimensional array."""
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    flattened = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.isfinite(flattened).all():
        raise ValueError("{} contains non-finite values.".format(name))
    return flattened


def _format_summary_value(value: Any) -> str:
    """Format one structured summary value for readable plain text."""
    serializable = _to_serializable(value)
    if isinstance(serializable, (dict, list)):
        return json.dumps(serializable, ensure_ascii=False)
    return str(serializable)


__all__ = [
    "ExperimentPaths",
    "append_training_log",
    "create_experiment_dir",
    "save_checkpoint",
    "save_config",
    "save_experiment_summary",
    "save_lct_parameters",
    "save_metrics",
    "save_prediction_results",
]
