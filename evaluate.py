"""Independent evaluation entry point for trained financial forecasters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import torch
from torch import nn

from src.data.dataset import create_dataloaders
from src.models.lstm_forecaster import LCTRieszLSTMForecaster
from src.utils.experiment_io import (
    ExperimentPaths,
    append_experiment_index,
    create_experiment_dir,
    save_experiment_summary,
    save_lct_parameters,
    save_metrics,
    save_prediction_results,
)
from src.utils.metrics import calculate_all_metrics
from src.utils.visualization import (
    plot_error_histogram,
    plot_prediction_curve,
    plot_residual_distribution,
)
from train import collect_predictions, evaluate, load_config, resolve_device, set_seed


PathLike = Union[str, Path]


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: PathLike,
    device: torch.device,
) -> Dict[str, Any]:
    """Load a checkpoint's model state and return its metadata."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError("Checkpoint file does not exist: {}".format(path))

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary.")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model_state_dict.")

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def run_evaluation(
    config_path: PathLike,
    checkpoint_path: PathLike,
) -> ExperimentPaths:
    """Load a trained model, evaluate the test set, and save new artifacts."""
    config = load_config(config_path)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    experiment_config = config["experiment"]

    csv_path = Path(data_config["csv_path"])
    if not csv_path.is_file():
        raise FileNotFoundError(
            "CSV file not found. Please place your OHLCV data at "
            "data/raw/sample.csv or update configs/lstm_return.yaml."
        )

    seed = int(training_config["seed"])
    set_seed(seed)
    device = resolve_device(training_config.get("device", "auto"))

    data_bundle = create_dataloaders(
        csv_path=str(csv_path),
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

    model = LCTRieszLSTMForecaster(
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        lstm_hidden_dim=int(model_config["lstm_hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        output_dim=int(model_config["output_dim"]),
        dropout=float(model_config["dropout"]),
        bidirectional=bool(model_config["bidirectional"]),
        use_lct_riesz=bool(model_config["use_lct_riesz"]),
        lct_alpha=float(model_config.get("lct_alpha", 1.0)),
        lct_m=float(model_config.get("lct_m", 1.0)),
        lct_q=float(model_config.get("lct_q", 0.0)),
        riesz_gamma=float(model_config.get("riesz_gamma", 1.0)),
        learnable_gamma=bool(model_config.get("learnable_gamma", False)),
        lct_gate_init=float(model_config.get("lct_gate_init", 0.0)),
    ).to(device)

    checkpoint = load_checkpoint(model, checkpoint_path, device)
    criterion = nn.MSELoss()
    test_loss = evaluate(
        model,
        data_bundle.test_loader,
        criterion,
        device,
    )
    y_true, y_pred = collect_predictions(
        model,
        data_bundle.test_loader,
        device,
        target_scaler=data_bundle.target_scaler,
    )

    include_directional_accuracy = str(data_config["target_type"]) in {
        "return",
        "log_return",
    }
    metrics = calculate_all_metrics(
        y_true,
        y_pred,
        include_directional_accuracy=include_directional_accuracy,
    )
    metrics["test_loss"] = test_loss

    paths = create_experiment_dir(
        outputs_root=experiment_config.get("outputs_root", "outputs"),
        checkpoints_root=experiment_config.get(
            "checkpoints_root",
            "checkpoints",
        ),
        prefix="evaluation",
        experiment_name=experiment_config.get("name"),
    )
    save_metrics(metrics, paths.metrics_path)
    save_prediction_results(y_true, y_pred, paths.prediction_results_path)

    lct_parameters = model.export_lct_parameters()
    if lct_parameters is not None:
        save_lct_parameters(lct_parameters, paths.lct_parameters_path)

    checkpoint_metrics = checkpoint.get("metrics", {})
    summary: Mapping[str, Any] = {
        "evaluation_name": experiment_config.get(
            "name",
            "unnamed_evaluation",
        ),
        "checkpoint_path": str(Path(checkpoint_path)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint_metrics,
        "dataset": str(csv_path),
        "target_type": data_config["target_type"],
        "device": str(device),
        "test_samples": len(data_bundle.test_dataset),
        "trainable_parameters": model.count_parameters(),
        "use_lct_riesz": model.use_lct_riesz,
        "metrics": metrics,
    }
    save_experiment_summary(summary, paths.summary_path)

    plot_prediction_curve(y_true, y_pred, paths.figures_dir)
    plot_residual_distribution(y_true, y_pred, paths.figures_dir)
    plot_error_histogram(y_true, y_pred, paths.figures_dir)
    append_experiment_index(
        outputs_root=experiment_config.get("outputs_root", "outputs"),
        run_dir=str(paths.experiment_dir),
        checkpoint_dir=str(paths.checkpoint_dir),
        prefix="evaluation",
        experiment_name=experiment_config.get("name", "unnamed_evaluation"),
        target_type=data_config["target_type"],
        model_type="lct_riesz_lstm" if model.use_lct_riesz else "plain_lstm",
        use_lct_riesz=model.use_lct_riesz,
        epochs=training_config.get("epochs"),
        best_epoch=checkpoint.get("epoch"),
        rmse=metrics.get("rmse"),
        mae=metrics.get("mae"),
        mse=metrics.get("mse"),
        mape=metrics.get("mape"),
        r2=metrics.get("r2"),
        directional_accuracy=metrics.get("directional_accuracy"),
        best_val_loss=checkpoint_metrics.get("val_loss"),
        test_loss=test_loss,
    )
    return paths


def main(
    config_path: PathLike = "configs/lstm_return.yaml",
    checkpoint_path: PathLike = None,
) -> ExperimentPaths:
    """Run independent evaluation for a required trained checkpoint."""
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for evaluation.")
    return run_evaluation(config_path, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LCT-Riesz LSTM checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="configs/lstm_return.yaml",
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best_model.pth or another compatible checkpoint.",
    )
    arguments = parser.parse_args()
    main(arguments.config, arguments.checkpoint)
