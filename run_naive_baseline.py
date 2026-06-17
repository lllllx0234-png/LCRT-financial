"""Naive financial forecasting baselines for close and return targets."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from src.data.dataset import chronological_split, load_ohlcv_csv
from src.utils.experiment_io import (
    ExperimentPaths,
    create_experiment_dir,
    save_config,
    save_experiment_summary,
    save_metrics,
    save_prediction_results,
)
from src.utils.metrics import calculate_all_metrics
from src.utils.visualization import (
    plot_error_histogram,
    plot_prediction_curve,
    plot_residual_distribution,
)
from train import load_config


PathLike = Union[str, Path]


def build_naive_predictions(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Construct naive test-set predictions on the original target scale."""
    data_config = config["data"]
    target_type = str(data_config["target_type"])
    sequence_length = int(data_config["sequence_length"])

    raw_data = load_ohlcv_csv(
        str(data_config["csv_path"]),
        feature_columns=data_config.get("feature_columns"),
    )
    _, _, raw_test = chronological_split(
        raw_data,
        train_ratio=float(data_config["train_ratio"]),
        val_ratio=float(data_config["val_ratio"]),
        test_ratio=float(data_config["test_ratio"]),
    )
    if len(raw_test) <= sequence_length:
        raise ValueError(
            "Test split length must be greater than sequence_length."
        )

    close_values = raw_test["Close"].to_numpy(dtype=np.float64)
    y_true = []
    y_pred = []
    for target_index in range(sequence_length, len(raw_test)):
        if target_type == "close":
            # Tomorrow close ~= today's close, using raw prices from the test split.
            y_true.append(close_values[target_index])
            y_pred.append(close_values[target_index - 1])
        elif target_type == "return":
            previous_close = close_values[target_index - 1]
            if previous_close == 0.0:
                raise ValueError("Cannot calculate return from a zero previous Close.")
            y_true.append(close_values[target_index] / previous_close - 1.0)
            y_pred.append(0.0)
        else:
            raise ValueError("target_type must be 'close' or 'return'.")

    return (
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )


def _naive_metadata(target_type: str) -> Dict[str, str]:
    """Return stable output labels for a naive baseline target type."""
    if target_type == "close":
        return {
            "experiment_name": "naive_last_close",
            "naive_rule": "last_close",
            "prediction_rule": "next Close = last Close in input window",
        }
    if target_type == "return":
        return {
            "experiment_name": "naive_zero_return",
            "naive_rule": "zero_return",
            "prediction_rule": "next return = 0",
        }
    raise ValueError("target_type must be 'close' or 'return'.")


def run_naive_baseline(
    config_path: PathLike = "configs/lstm_close_lct.yaml",
) -> ExperimentPaths:
    """Run a naive baseline and save metrics, predictions, summary, and plots."""
    config = load_config(config_path)
    data_config = config["data"]
    experiment_config = config["experiment"]
    target_type = str(data_config["target_type"])
    metadata = _naive_metadata(target_type)

    y_true, y_pred = build_naive_predictions(config)
    include_directional_accuracy = target_type == "return"
    metrics = calculate_all_metrics(
        y_true,
        y_pred,
        include_directional_accuracy=include_directional_accuracy,
    )

    paths = create_experiment_dir(
        outputs_root=experiment_config.get("outputs_root", "outputs"),
        checkpoints_root=experiment_config.get("checkpoints_root", "checkpoints"),
        prefix="naive",
    )

    saved_config = copy.deepcopy(config)
    saved_config.setdefault("model", {})
    saved_config.setdefault("experiment", {})
    saved_config["model"]["type"] = "naive"
    saved_config["experiment"]["name"] = metadata["experiment_name"]
    saved_config["naive_rule"] = metadata["naive_rule"]
    saved_config["runtime"] = {
        "baseline": "naive",
        "torch_version": torch.__version__,
        "target_scale": "raw_close" if target_type == "close" else "return",
    }
    save_config(saved_config, paths.config_path)
    save_metrics(metrics, paths.metrics_path)
    save_prediction_results(y_true, y_pred, paths.prediction_results_path)

    summary = {
        "experiment_name": metadata["experiment_name"],
        "baseline": "naive",
        "model.type": "naive",
        "naive_rule": metadata["naive_rule"],
        "prediction_rule": metadata["prediction_rule"],
        "source_experiment_name": experiment_config.get("name", "naive_baseline"),
        "dataset": str(data_config["csv_path"]),
        "target_type": target_type,
        "sequence_length": data_config["sequence_length"],
        "test_samples": int(y_true.shape[0]),
        "metrics": metrics,
    }
    save_experiment_summary(summary, paths.summary_path)

    plot_prediction_curve(y_true, y_pred, paths.figures_dir)
    plot_residual_distribution(y_true, y_pred, paths.figures_dir)
    plot_error_histogram(y_true, y_pred, paths.figures_dir)
    return paths


def main() -> None:
    """Command-line entry point for the naive baseline."""
    parser = argparse.ArgumentParser(
        description="Run a naive financial forecasting baseline.",
    )
    parser.add_argument(
        "--config",
        default="configs/lstm_close_lct.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()
    run_naive_baseline(arguments.config)


if __name__ == "__main__":
    main()
