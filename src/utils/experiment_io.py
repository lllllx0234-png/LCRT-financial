"""Reproducible experiment directory and artifact persistence utilities."""

from __future__ import annotations

import csv
import json
import re
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


@dataclass(frozen=True)
class ExperimentClassification:
    """Task and run-type labels used for hierarchical experiment storage."""

    task_name: str
    run_type: str


def create_experiment_dir(
    outputs_root: PathLike = "outputs",
    checkpoints_root: PathLike = "checkpoints",
    timestamp: Optional[datetime] = None,
    prefix: str = "experiment",
    experiment_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[PathLike] = None,
) -> ExperimentPaths:
    """Create unique hierarchical output and checkpoint directories for one run."""
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
    classification = classify_experiment_run(
        prefix=normalized_prefix,
        experiment_name=experiment_name,
        config=config,
        config_path=config_path,
    )
    output_parent = outputs_directory / classification.task_name / classification.run_type
    checkpoint_parent = (
        checkpoints_directory / classification.task_name / classification.run_type
    )
    output_parent.mkdir(parents=True, exist_ok=True)
    checkpoint_parent.mkdir(parents=True, exist_ok=True)

    run_id = _find_unique_run_id(
        run_time.strftime("%Y%m%d_%H%M%S"),
        output_parent,
        checkpoint_parent,
    )

    experiment_dir = output_parent / run_id
    checkpoint_dir = checkpoint_parent / run_id
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


EXPERIMENT_INDEX_FIELDS = [
    "run_dir",
    "checkpoint_dir",
    "prefix",
    "experiment_name",
    "target_type",
    "model_type",
    "use_lct_riesz",
    "naive_rule",
    "epochs",
    "best_epoch",
    "rmse",
    "mae",
    "mse",
    "mape",
    "r2",
    "directional_accuracy",
    "best_val_loss",
    "test_loss",
    "created_at",
]


def append_experiment_index(
    outputs_root: PathLike = "outputs",
    **fields: Any,
) -> Path:
    """Append one run summary row to outputs/experiment_index.csv."""
    index_path = Path(outputs_root) / "experiment_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: "" for field in EXPERIMENT_INDEX_FIELDS}
    for key, value in fields.items():
        if key in row:
            row[key] = _to_index_value(value)
    if not row["created_at"]:
        row["created_at"] = datetime.now().isoformat(timespec="seconds")

    file_exists = index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=EXPERIMENT_INDEX_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return index_path


def classify_experiment_run(
    prefix: str = "experiment",
    experiment_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[PathLike] = None,
) -> ExperimentClassification:
    """Infer the task and run type for hierarchical experiment outputs."""
    normalized_prefix = _sanitize_experiment_name(prefix)
    safe_name = _sanitize_experiment_name(experiment_name)
    config_stem = _sanitize_experiment_name(Path(config_path).stem if config_path else "")
    text_parts = [normalized_prefix, safe_name, config_stem]

    data_config: Mapping[str, Any] = {}
    model_config: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        data_section = config.get("data", {})
        model_section = config.get("model", {})
        experiment_section = config.get("experiment", {})
        if isinstance(data_section, Mapping):
            data_config = data_section
            text_parts.append(_sanitize_experiment_name(data_config.get("target_type")))
        if isinstance(model_section, Mapping):
            model_config = model_section
            text_parts.append(_sanitize_experiment_name(model_config.get("type")))
        if isinstance(experiment_section, Mapping):
            text_parts.append(_sanitize_experiment_name(experiment_section.get("name")))

    inference_text = "_".join(part for part in text_parts if part)
    target_type = str(data_config.get("target_type", "")).strip().lower()
    use_lct_riesz = bool(model_config.get("use_lct_riesz", False))
    has_derived_features = bool(data_config.get("derived_features"))

    if normalized_prefix == "naive" or "naive" in inference_text:
        return ExperimentClassification(
            task_name="naive",
            run_type=_infer_naive_run_type(inference_text, target_type),
        )

    if "volatility_5" in inference_text or target_type == "volatility_5":
        return ExperimentClassification(
            task_name="volatility_5",
            run_type=_infer_model_run_type(
                inference_text,
                use_lct_riesz,
                has_derived_features,
            ),
        )

    if "log_return" in inference_text or target_type == "log_return":
        return ExperimentClassification(
            task_name="log_return",
            run_type=_infer_model_run_type(
                inference_text,
                use_lct_riesz,
                has_derived_features,
            ),
        )

    return ExperimentClassification(task_name="archive", run_type="unknown")


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


def _find_unique_run_id(
    base_run_id: str,
    outputs_parent: Path,
    checkpoints_parent: Path,
) -> str:
    """Find a timestamp run ID unused by both output and checkpoint parents."""
    candidate = base_run_id
    suffix = 1
    while (outputs_parent / candidate).exists() or (
        checkpoints_parent / candidate
    ).exists():
        candidate = "{}_{:02d}".format(base_run_id, suffix)
        suffix += 1
    return candidate


def _sanitize_experiment_name(
    experiment_name: Optional[Any],
    max_length: int = 80,
) -> str:
    """Convert an experiment name into a compact filesystem-safe suffix."""
    if experiment_name is None:
        return ""
    normalized = str(experiment_name).strip().lower()
    if not normalized:
        return ""
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_-")
    return normalized[:max_length].strip("_-")


def _infer_model_run_type(
    inference_text: str,
    use_lct_riesz: bool,
    has_derived_features: bool,
) -> str:
    """Infer a deep model run type from names first, then model flags."""
    for run_type in (
        "dual_branch_signal_features",
        "dual_branch_features",
        "dual_branch_lct_riesz_lstm",
        "lct_signal_features",
        "baseline_signal_features",
        "lct_riesz_features",
        "baseline_features",
        "lct_riesz",
        "baseline",
    ):
        if run_type in inference_text:
            return run_type
    if has_derived_features or "features" in inference_text:
        return "lct_riesz_features" if use_lct_riesz else "baseline_features"
    return "lct_riesz" if use_lct_riesz else "baseline"


def _infer_naive_run_type(inference_text: str, target_type: str) -> str:
    """Infer the naive-baseline storage bucket for a target type."""
    if "zero_log_return" in inference_text or target_type == "log_return":
        return "zero_log_return"
    if "historical_volatility_5" in inference_text or target_type == "volatility_5":
        return "historical_volatility_5"
    return "legacy"


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


def _to_index_value(value: Any) -> str:
    """Format a value for one CSV experiment-index cell."""
    if value is None:
        return ""
    serializable = _to_serializable(value)
    if isinstance(serializable, (dict, list)):
        return json.dumps(serializable, ensure_ascii=False)
    return str(serializable)


__all__ = [
    "EXPERIMENT_INDEX_FIELDS",
    "ExperimentClassification",
    "ExperimentPaths",
    "append_experiment_index",
    "append_training_log",
    "classify_experiment_run",
    "create_experiment_dir",
    "save_checkpoint",
    "save_config",
    "save_experiment_summary",
    "save_lct_parameters",
    "save_metrics",
    "save_prediction_results",
]
