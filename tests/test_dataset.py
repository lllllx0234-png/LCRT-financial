"""Tests for chronological OHLCV preprocessing and sliding windows."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import (
    DEFAULT_FEATURE_COLUMNS,
    build_sliding_windows,
    chronological_split,
    create_dataloaders,
    fit_transform_scaler,
    load_ohlcv_csv,
)


class FinancialDatasetTest(unittest.TestCase):
    """Verify loading, leakage prevention, targets, and data loaders."""

    def setUp(self) -> None:
        """Create a shuffled 120-row OHLCV CSV in a temporary directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "ohlcv.csv"

        rows = 120
        dates = pd.date_range("2024-01-01", periods=rows, freq="D")
        close = 100.0 + np.arange(rows, dtype=np.float64)
        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000.0 + np.arange(rows) * 10.0,
            }
        )
        data.sample(frac=1.0, random_state=42).to_csv(
            self.csv_path,
            index=False,
        )

    def tearDown(self) -> None:
        """Remove the temporary CSV and directory."""
        self.temp_dir.cleanup()

    def test_csv_loading_sorts_dates(self) -> None:
        """Read all required columns and restore chronological order."""
        data = load_ohlcv_csv(str(self.csv_path))
        self.assertEqual(len(data), 120)
        self.assertTrue(data["Date"].is_monotonic_increasing)
        self.assertEqual(data.iloc[0]["Date"], pd.Timestamp("2024-01-01"))

    def test_chronological_split_and_train_only_scaler_fit(self) -> None:
        """Keep partitions ordered and fit statistics from train only."""
        data = load_ohlcv_csv(str(self.csv_path))
        train, val, test = chronological_split(data)

        self.assertLess(train["Date"].max(), val["Date"].min())
        self.assertLess(val["Date"].max(), test["Date"].min())
        self.assertEqual((len(train), len(val), len(test)), (84, 18, 18))

        scaled_train, scaled_val, scaled_test, scaler = fit_transform_scaler(
            train,
            val,
            test,
        )
        np.testing.assert_allclose(
            scaler.mean_,
            train[DEFAULT_FEATURE_COLUMNS].mean().to_numpy(),
        )
        np.testing.assert_allclose(
            scaled_train[DEFAULT_FEATURE_COLUMNS].mean().to_numpy(),
            np.zeros(5),
            atol=1e-6,
        )
        self.assertGreater(
            scaled_val[DEFAULT_FEATURE_COLUMNS].mean().mean(),
            0.0,
        )
        self.assertGreater(
            scaled_test[DEFAULT_FEATURE_COLUMNS].mean().mean(),
            scaled_val[DEFAULT_FEATURE_COLUMNS].mean().mean(),
        )

    def test_close_and_return_sliding_windows(self) -> None:
        """Build correctly shaped next-close and next-return targets."""
        data = load_ohlcv_csv(str(self.csv_path))
        close_x, close_y, close_dates = build_sliding_windows(
            data,
            sequence_length=10,
            target_type="close",
        )
        return_x, return_y, return_dates = build_sliding_windows(
            data,
            sequence_length=10,
            target_type="return",
        )

        self.assertEqual(close_x.shape, (110, 10, 5))
        self.assertEqual(close_y.shape, (110, 1))
        self.assertEqual(return_x.shape, (110, 10, 5))
        self.assertEqual(return_y.shape, (110, 1))
        self.assertEqual(close_dates, return_dates)
        self.assertAlmostEqual(float(close_y[0, 0]), 110.0)
        self.assertAlmostEqual(float(return_y[0, 0]), 110.0 / 109.0 - 1.0)

    def test_create_dataloaders_batch_shapes_and_targets(self) -> None:
        """Create chronological close and return data-loader pipelines."""
        close_bundle = create_dataloaders(
            str(self.csv_path),
            sequence_length=10,
            batch_size=8,
            target_type="close",
        )
        batch_x, batch_y = next(iter(close_bundle.train_loader))

        self.assertEqual(tuple(batch_x.shape), (8, 10, 5))
        self.assertEqual(tuple(batch_y.shape), (8, 1))
        self.assertEqual(
            (
                len(close_bundle.train_dataset),
                len(close_bundle.val_dataset),
                len(close_bundle.test_dataset),
            ),
            (74, 8, 8),
        )
        self.assertLess(
            close_bundle.train_dataset.target_dates[-1],
            close_bundle.val_dataset.target_dates[0],
        )
        self.assertLess(
            close_bundle.val_dataset.target_dates[-1],
            close_bundle.test_dataset.target_dates[0],
        )

        return_bundle = create_dataloaders(
            str(self.csv_path),
            sequence_length=10,
            batch_size=8,
            target_type="return",
        )
        return_x, return_y = next(iter(return_bundle.train_loader))
        self.assertEqual(tuple(return_x.shape), (8, 10, 5))
        self.assertEqual(tuple(return_y.shape), (8, 1))
        self.assertTrue(torch.isfinite(return_y).all())


if __name__ == "__main__":
    unittest.main()
