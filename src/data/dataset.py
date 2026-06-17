"""Financial time-series loading, preprocessing, and window construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


DEFAULT_FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
REQUIRED_OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
VALID_TARGET_TYPES = {"close", "return", "log_return"}


class FinancialTimeSeriesDataset(Dataset):
    """Store sliding windows and one-step financial prediction targets."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        target_dates: Optional[Sequence[pd.Timestamp]] = None,
    ) -> None:
        """Initialize a dataset from precomputed windows and targets."""
        if features.ndim != 3:
            raise ValueError(
                "features must have shape (samples, sequence_length, features)."
            )
        if targets.ndim != 2 or targets.shape[1] != 1:
            raise ValueError("targets must have shape (samples, 1).")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must contain equal samples.")
        if target_dates is not None and len(target_dates) != features.shape[0]:
            raise ValueError("target_dates must contain one date per sample.")

        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.target_dates = (
            tuple(pd.Timestamp(date) for date in target_dates)
            if target_dates is not None
            else None
        )

    def __len__(self) -> int:
        """Return the number of sliding-window samples."""
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one feature window and its one-step target."""
        return self.features[index], self.targets[index]


@dataclass(frozen=True)
class FinancialDataLoaders:
    """Bundle datasets, data loaders, scaler, and preprocessing metadata."""

    train_dataset: FinancialTimeSeriesDataset
    val_dataset: FinancialTimeSeriesDataset
    test_dataset: FinancialTimeSeriesDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    target_scaler: Optional[StandardScaler]
    preprocessing_config: Dict[str, object]


def load_ohlcv_csv(
    csv_path: str,
    feature_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Read, validate, and chronologically sort an OHLCV CSV file."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError("CSV file does not exist: {}".format(path))

    selected_features = _validate_feature_columns(feature_columns)
    data = pd.read_csv(path)

    missing_required = [
        column for column in REQUIRED_OHLCV_COLUMNS if column not in data.columns
    ]
    if missing_required:
        raise ValueError(
            "CSV is missing required columns: {}".format(missing_required)
        )

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    numeric_columns = sorted(set(DEFAULT_FEATURE_COLUMNS + selected_features))
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    checked_columns = ["Date"] + numeric_columns
    if data[checked_columns].isnull().any().any():
        missing_counts = data[checked_columns].isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError("CSV contains missing values: {}".format(missing_counts))

    duplicated_dates = data.loc[data["Date"].duplicated(keep=False), "Date"]
    if not duplicated_dates.empty:
        duplicates = duplicated_dates.dt.strftime("%Y-%m-%d").unique().tolist()
        raise ValueError("CSV contains duplicate dates: {}".format(duplicates))

    return data.sort_values("Date").reset_index(drop=True)


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a sorted time series into contiguous train, validation, and test sets."""
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    if len(data) < 3:
        raise ValueError("At least three rows are required for chronological split.")
    if "Date" not in data.columns:
        raise ValueError("data must contain a Date column.")
    if not data["Date"].is_monotonic_increasing:
        raise ValueError("data must be sorted by Date before splitting.")

    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * val_ratio)
    if train_end == 0 or val_end == train_end or val_end >= len(data):
        raise ValueError("Split ratios produce an empty data partition.")

    train = data.iloc[:train_end].copy().reset_index(drop=True)
    val = data.iloc[train_end:val_end].copy().reset_index(drop=True)
    test = data.iloc[val_end:].copy().reset_index(drop=True)
    return train, val, test


def fit_transform_scaler(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a scaler on train features and transform all chronological splits."""
    selected_features = _validate_feature_columns(feature_columns)
    fitted_scaler = scaler if scaler is not None else StandardScaler()
    fitted_scaler.fit(train_data[selected_features].to_numpy(dtype=np.float64))

    transformed_splits = []
    for split in (train_data, val_data, test_data):
        transformed = split.copy()
        transformed.loc[:, selected_features] = fitted_scaler.transform(
            split[selected_features].to_numpy(dtype=np.float64)
        )
        transformed_splits.append(transformed)

    return (
        transformed_splits[0],
        transformed_splits[1],
        transformed_splits[2],
        fitted_scaler,
    )


def build_sliding_windows(
    data: pd.DataFrame,
    sequence_length: int,
    feature_columns: Optional[Sequence[str]] = None,
    target_type: str = "close",
    target_close: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Construct one-step-ahead windows from one chronological data split."""
    selected_features = _validate_feature_columns(feature_columns)
    _validate_target_type(target_type)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if len(data) <= sequence_length:
        raise ValueError(
            "data length must be greater than sequence_length for one-step targets."
        )
    if "Date" not in data.columns:
        raise ValueError("data must contain a Date column.")

    feature_values = data[selected_features].to_numpy(dtype=np.float32)
    if target_close is None:
        close_values = data["Close"].to_numpy(dtype=np.float64)
    else:
        close_values = np.asarray(target_close, dtype=np.float64)
        if close_values.shape != (len(data),):
            raise ValueError("target_close must contain one value per data row.")

    windows = []
    targets = []
    target_dates = []
    for target_index in range(sequence_length, len(data)):
        windows.append(feature_values[target_index - sequence_length : target_index])
        if target_type == "close":
            target = close_values[target_index]
        elif target_type == "return":
            previous_close = close_values[target_index - 1]
            if previous_close == 0.0:
                raise ValueError("Cannot calculate return from a zero previous Close.")
            target = close_values[target_index] / previous_close - 1.0
        else:
            previous_close = close_values[target_index - 1]
            current_close = close_values[target_index]
            if previous_close <= 0.0 or current_close <= 0.0:
                raise ValueError(
                    "Cannot calculate log_return because Close values must be positive."
                )
            target = np.log(current_close / previous_close)
        targets.append([target])
        target_dates.append(pd.Timestamp(data.iloc[target_index]["Date"]))

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        target_dates,
    )


def create_dataloaders(
    csv_path: str,
    sequence_length: int,
    batch_size: int,
    feature_columns: Optional[Sequence[str]] = None,
    target_type: str = "close",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle_train: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> FinancialDataLoaders:
    """Create chronological datasets and data loaders without scaler leakage."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    selected_features = _validate_feature_columns(feature_columns)
    _validate_target_type(target_type)
    raw_data = load_ohlcv_csv(csv_path, selected_features)
    raw_train, raw_val, raw_test = chronological_split(
        raw_data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_data, val_data, test_data, scaler = fit_transform_scaler(
        raw_train,
        raw_val,
        raw_test,
        selected_features,
    )
    target_scaler: Optional[StandardScaler] = None
    if target_type == "close":
        target_scaler = StandardScaler()
        target_scaler.fit(raw_train[["Close"]].to_numpy(dtype=np.float64))

    datasets = []
    for transformed, raw in (
        (train_data, raw_train),
        (val_data, raw_val),
        (test_data, raw_test),
    ):
        features, targets, target_dates = build_sliding_windows(
            transformed,
            sequence_length=sequence_length,
            feature_columns=selected_features,
            target_type=target_type,
            target_close=raw["Close"].to_numpy(dtype=np.float64),
        )
        if target_scaler is not None:
            targets = target_scaler.transform(
                targets.astype(np.float64)
            ).astype(np.float32)
        datasets.append(
            FinancialTimeSeriesDataset(features, targets, target_dates)
        )

    train_dataset, val_dataset, test_dataset = datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    preprocessing_config = {
        "csv_path": str(Path(csv_path)),
        "feature_columns": list(selected_features),
        "target_type": target_type,
        "sequence_length": sequence_length,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "scaler": scaler.__class__.__name__,
        "target_scaler_enabled": target_scaler is not None,
        "target_scaler": (
            target_scaler.__class__.__name__
            if target_scaler is not None
            else None
        ),
    }
    return FinancialDataLoaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        target_scaler=target_scaler,
        preprocessing_config=preprocessing_config,
    )


def _validate_feature_columns(
    feature_columns: Optional[Sequence[str]],
) -> List[str]:
    """Return a validated, duplicate-free feature column list."""
    selected = (
        list(DEFAULT_FEATURE_COLUMNS)
        if feature_columns is None
        else list(feature_columns)
    )
    if not selected:
        raise ValueError("feature_columns cannot be empty.")
    if len(selected) != len(set(selected)):
        raise ValueError("feature_columns cannot contain duplicates.")
    return selected


def _validate_target_type(target_type: str) -> None:
    """Validate the supported one-step prediction target."""
    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(
            "target_type must be one of {}.".format(sorted(VALID_TARGET_TYPES))
        )


def _validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Validate positive chronological split ratios that sum to one."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("All split ratios must be positive.")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.")


__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "FinancialDataLoaders",
    "FinancialTimeSeriesDataset",
    "build_sliding_windows",
    "chronological_split",
    "create_dataloaders",
    "fit_transform_scaler",
    "load_ohlcv_csv",
]
