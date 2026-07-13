"""Training entry point for LCT-Riesz enhanced financial forecasting."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import create_dataloaders
from src.models.dual_branch_lct_lstm import DualBranchLCTRieszLSTMForecaster
from src.models.lstm_forecaster import LCTRieszLSTMForecaster
from src.models.residual_lct_lstm import ResidualAuxiliaryLCTRieszLSTMForecaster
from src.models.tcn_forecaster import TCNForecaster
from src.utils.experiment_io import (
    ExperimentPaths,
    append_experiment_index,
    append_training_log,
    create_experiment_dir,
    save_checkpoint,
    save_config,
    save_experiment_summary,
    save_lct_parameters,
    save_metrics,
    save_prediction_results,
)
from src.utils.metrics import calculate_all_metrics
from src.utils.visualization import (
    plot_error_histogram,
    plot_lct_parameter_history,
    plot_loss_curves,
    plot_prediction_curve,
    plot_residual_distribution,
)


PathLike = Union[str, Path]


def load_config(config_path: PathLike) -> Dict[str, Any]:
    """Load and validate the top-level YAML experiment configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError("Configuration file does not exist: {}".format(path))

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    required_sections = {"data", "model", "training", "experiment"}
    missing_sections = required_sections.difference(config)
    if missing_sections:
        raise ValueError(
            "Configuration is missing sections: {}".format(
                sorted(missing_sections)
            )
        )
    return config


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_config: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise use the requested device."""
    normalized = str(device_config).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_model(model_config: Mapping[str, Any]) -> nn.Module:
    """Build the configured forecasting model while preserving legacy configs."""
    model_type = str(model_config.get("type", "")).strip().lower()
    if model_type == "plain_tcn":
        return TCNForecaster(
            input_dim=int(model_config["input_dim"]),
            channels=[int(channel) for channel in model_config["channels"]],
            kernel_size=int(model_config["kernel_size"]),
            dropout=float(model_config["dropout"]),
            output_dim=int(model_config.get("output_dim", 1)),
        )

    if model_type in {"", "lct_riesz_lstm", "plain_lstm"}:
        return LCTRieszLSTMForecaster(
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
        )

    if model_type == "dual_branch_lct_riesz_lstm":
        return DualBranchLCTRieszLSTMForecaster(
            input_dim=int(model_config["input_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            lstm_hidden_dim=int(model_config["lstm_hidden_dim"]),
            signal_feature_indices=model_config["signal_feature_indices"],
            num_layers=int(model_config["num_layers"]),
            output_dim=int(model_config["output_dim"]),
            dropout=float(model_config["dropout"]),
            bidirectional=bool(model_config["bidirectional"]),
            use_lct_riesz=bool(model_config.get("use_lct_riesz", True)),
            spectral_hidden_dim=(
                int(model_config["spectral_hidden_dim"])
                if "spectral_hidden_dim" in model_config
                else None
            ),
            fusion_gate_init=float(model_config.get("fusion_gate_init", -3.0)),
            lct_alpha=float(model_config.get("lct_alpha", 1.0)),
            lct_m=float(model_config.get("lct_m", 1.0)),
            lct_q=float(model_config.get("lct_q", 0.0)),
            riesz_gamma=float(model_config.get("riesz_gamma", 1.0)),
            learnable_gamma=bool(model_config.get("learnable_gamma", False)),
            lct_gate_init=float(model_config.get("lct_gate_init", 1.0)),
        )

    if model_type == "residual_auxiliary_lct_riesz_lstm":
        return ResidualAuxiliaryLCTRieszLSTMForecaster(
            input_dim=int(model_config["input_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            lstm_hidden_dim=int(model_config["lstm_hidden_dim"]),
            signal_feature_indices=model_config["signal_feature_indices"],
            num_layers=int(model_config["num_layers"]),
            output_dim=int(model_config["output_dim"]),
            dropout=float(model_config["dropout"]),
            bidirectional=bool(model_config["bidirectional"]),
            use_lct_riesz=bool(model_config.get("use_lct_riesz", True)),
            spectral_hidden_dim=(
                int(model_config["spectral_hidden_dim"])
                if "spectral_hidden_dim" in model_config
                else None
            ),
            residual_scale_init=float(
                model_config.get("residual_scale_init", 0.0)
            ),
            lct_alpha=float(model_config.get("lct_alpha", 1.0)),
            lct_m=float(model_config.get("lct_m", 1.0)),
            lct_q=float(model_config.get("lct_q", 0.0)),
            riesz_gamma=float(model_config.get("riesz_gamma", 1.0)),
            learnable_gamma=bool(model_config.get("learnable_gamma", False)),
            lct_gate_init=float(model_config.get("lct_gate_init", 1.0)),
        )

    raise ValueError("Unsupported model.type: {}".format(model_type))


def model_type_for_index(model_config: Mapping[str, Any], model: nn.Module) -> str:
    """Return the stable model_type label used in experiment_index.csv."""
    configured_type = str(model_config.get("type", "")).strip().lower()
    if configured_type:
        return configured_type
    use_lct_riesz = bool(getattr(model, "use_lct_riesz", False))
    return "lct_riesz_lstm" if use_lct_riesz else "plain_lstm"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader contains no samples.")
    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate a model and return sample-weighted mean loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(features)
        loss = criterion(predictions, targets)

        batch_size = features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Evaluation DataLoader contains no samples.")
    return total_loss / total_samples


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: Any = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect flattened true and predicted values from a DataLoader."""
    model.eval()
    true_batches: List[np.ndarray] = []
    predicted_batches: List[np.ndarray] = []

    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        predictions = model(features)
        true_batches.append(targets.detach().cpu().numpy())
        predicted_batches.append(predictions.detach().cpu().numpy())

    if not true_batches:
        raise ValueError("Prediction DataLoader contains no samples.")
    y_true = np.concatenate(true_batches, axis=0).reshape(-1)
    y_pred = np.concatenate(predicted_batches, axis=0).reshape(-1)
    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(
            y_true.reshape(-1, 1)
        ).reshape(-1)
        y_pred = target_scaler.inverse_transform(
            y_pred.reshape(-1, 1)
        ).reshape(-1)
    return y_true, y_pred


def main(
    config_path: PathLike = "experiments/lstm/configs/lstm_return.yaml",
) -> ExperimentPaths:
    """Run training, validation, test evaluation, and artifact generation."""
    config = load_config(config_path)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    experiment_config = config["experiment"]

    csv_path = Path(data_config["csv_path"])
    if not csv_path.is_file():
        raise FileNotFoundError(
            "CSV file not found. Please place your OHLCV data at "
            "data/raw/sample.csv or update experiments/lstm/configs/lstm_return.yaml."
        )

    seed = int(training_config["seed"])
    set_seed(seed)
    device = resolve_device(training_config.get("device", "auto"))

    paths = create_experiment_dir(
        outputs_root=experiment_config.get("outputs_root", "outputs"),
        checkpoints_root=experiment_config.get(
            "checkpoints_root",
            "checkpoints",
        ),
        experiment_name=experiment_config.get("name"),
        config=config,
        config_path=config_path,
    )

    saved_config = copy.deepcopy(config)
    saved_config["runtime"] = {
        "resolved_device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": seed,
    }
    save_config(saved_config, paths.config_path)

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
        shuffle_train=bool(data_config.get("shuffle_train", False)),
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_config).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    train_losses: List[float] = []
    val_losses: List[float] = []
    lct_parameter_history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs = int(training_config["epochs"])

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            data_bundle.train_loader,
            optimizer,
            criterion,
            device,
        )
        val_loss = evaluate(
            model,
            data_bundle.val_loader,
            criterion,
            device,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lct_parameters = model.export_lct_parameters()
        log_row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        }
        if lct_parameters is not None:
            history_row = {
                "epoch": float(epoch),
                "alpha": lct_parameters["alpha"],
                "m": lct_parameters["m"],
                "q": lct_parameters["q"],
                "gamma": lct_parameters["gamma"],
            }
            lct_parameter_history.append(history_row)
            log_row.update(
                {
                    "alpha": lct_parameters["alpha"],
                    "m": lct_parameters["m"],
                    "q": lct_parameters["q"],
                    "gamma": lct_parameters["gamma"],
                }
            )
        append_training_log(log_row, paths.training_log_path)

        checkpoint_metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        save_checkpoint(
            model,
            optimizer,
            epoch,
            checkpoint_metrics,
            paths.latest_model_path,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                model,
                optimizer,
                epoch,
                checkpoint_metrics,
                paths.best_model_path,
            )

    best_checkpoint = torch.load(
        paths.best_model_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

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
    metrics["best_val_loss"] = best_val_loss
    metrics["best_epoch"] = float(best_epoch)

    save_metrics(metrics, paths.metrics_path)
    save_prediction_results(y_true, y_pred, paths.prediction_results_path)

    final_lct_parameters = model.export_lct_parameters()
    if final_lct_parameters is not None:
        save_lct_parameters(final_lct_parameters, paths.lct_parameters_path)

    summary: Mapping[str, Any] = {
        "experiment_name": experiment_config.get("name", "unnamed_experiment"),
        "dataset": str(csv_path),
        "target_type": data_config["target_type"],
        "device": str(device),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "train_samples": len(data_bundle.train_dataset),
        "val_samples": len(data_bundle.val_dataset),
        "test_samples": len(data_bundle.test_dataset),
        "trainable_parameters": model.count_parameters(),
        "model_type": model_type_for_index(model_config, model),
        "use_lct_riesz": model.use_lct_riesz,
        "metrics": metrics,
    }
    save_experiment_summary(summary, paths.summary_path)

    plot_loss_curves(train_losses, val_losses, paths.figures_dir)
    plot_prediction_curve(y_true, y_pred, paths.figures_dir)
    plot_residual_distribution(y_true, y_pred, paths.figures_dir)
    plot_error_histogram(y_true, y_pred, paths.figures_dir)
    if lct_parameter_history:
        plot_lct_parameter_history(
            lct_parameter_history,
            paths.figures_dir,
        )

    append_experiment_index(
        outputs_root=experiment_config.get("outputs_root", "outputs"),
        run_dir=str(paths.experiment_dir),
        checkpoint_dir=str(paths.checkpoint_dir),
        prefix="experiment",
        experiment_name=experiment_config.get("name", "unnamed_experiment"),
        target_type=data_config["target_type"],
        model_type=model_type_for_index(model_config, model),
        use_lct_riesz=model.use_lct_riesz,
        epochs=epochs,
        best_epoch=best_epoch,
        rmse=metrics.get("rmse"),
        mae=metrics.get("mae"),
        mse=metrics.get("mse"),
        mape=metrics.get("mape"),
        r2=metrics.get("r2"),
        directional_accuracy=metrics.get("directional_accuracy"),
        best_val_loss=best_val_loss,
        test_loss=test_loss,
    )

    return paths


if __name__ == "__main__":
    main()
