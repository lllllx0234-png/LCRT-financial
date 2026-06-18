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
    SUPPORTED_DERIVED_FEATURES,
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

    def test_derived_features_use_current_and_past_data_only(self) -> None:
        """Create supported derived features without future information."""
        derived_features = [
            "log_return",
            "abs_log_return",
            "high_low_range",
            "close_open_return",
            "rolling_vol_5",
            "rolling_vol_10",
            "rolling_vol_20",
            "volume_change",
        ]
        feature_columns = DEFAULT_FEATURE_COLUMNS + derived_features
        data = load_ohlcv_csv(
            str(self.csv_path),
            feature_columns=feature_columns,
            derived_features=derived_features,
        )

        self.assertTrue(set(derived_features).issubset(SUPPORTED_DERIVED_FEATURES))
        self.assertEqual(data.iloc[0]["log_return"], 0.0)
        self.assertEqual(data.iloc[0]["volume_change"], 0.0)
        self.assertAlmostEqual(
            float(data.iloc[1]["log_return"]),
            np.log(101.0 / 100.0),
        )
        self.assertAlmostEqual(
            float(data.iloc[1]["abs_log_return"]),
            abs(np.log(101.0 / 100.0)),
        )
        self.assertAlmostEqual(float(data.iloc[0]["high_low_range"]), 2.0 / 100.0)
        self.assertAlmostEqual(
            float(data.iloc[0]["close_open_return"]),
            100.0 / 99.5 - 1.0,
        )
        self.assertAlmostEqual(
            float(data.iloc[1]["volume_change"]),
            np.log1p(1010.0) - np.log1p(1000.0),
        )

        historical_returns = np.array([0.0, np.log(101.0 / 100.0)])
        self.assertAlmostEqual(
            float(data.iloc[1]["rolling_vol_5"]),
            float(np.std(historical_returns, ddof=0)),
        )

    def test_volume_change_allows_zero_volume_and_uses_log1p_difference(self) -> None:
        """Calculate finite log-volume differences when volume can be zero."""
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=4, freq="D"),
                "Open": [10.0, 11.0, 12.0, 13.0],
                "High": [11.0, 12.0, 13.0, 14.0],
                "Low": [9.0, 10.0, 11.0, 12.0],
                "Close": [10.5, 11.5, 12.5, 13.5],
                "Volume": [100.0, 0.0, 50.0, 0.0],
            }
        )
        csv_path = Path(self.temp_dir.name) / "zero_volume.csv"
        data.to_csv(csv_path, index=False)

        loaded = load_ohlcv_csv(
            str(csv_path),
            feature_columns=["Volume", "volume_change"],
            derived_features=["volume_change"],
        )

        expected = np.array(
            [
                0.0,
                np.log1p(0.0) - np.log1p(100.0),
                np.log1p(50.0) - np.log1p(0.0),
                np.log1p(0.0) - np.log1p(50.0),
            ]
        )
        np.testing.assert_allclose(
            loaded["volume_change"].to_numpy(dtype=np.float64),
            expected,
        )
        self.assertTrue(np.isfinite(loaded["volume_change"]).all())

    def test_volume_change_rejects_negative_volume(self) -> None:
        """Reject negative volume values for log-volume differences."""
        data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
                "Open": [10.0, 11.0, 12.0],
                "High": [11.0, 12.0, 13.0],
                "Low": [9.0, 10.0, 11.0],
                "Close": [10.5, 11.5, 12.5],
                "Volume": [100.0, -1.0, 50.0],
            }
        )
        csv_path = Path(self.temp_dir.name) / "negative_volume.csv"
        data.to_csv(csv_path, index=False)

        with self.assertRaisesRegex(ValueError, "Volume values must be non-negative"):
            load_ohlcv_csv(
                str(csv_path),
                feature_columns=["Volume", "volume_change"],
                derived_features=["volume_change"],
            )

    def test_derived_features_reject_invalid_names(self) -> None:
        """Reject unknown derived feature names before preprocessing."""
        with self.assertRaisesRegex(ValueError, "Unsupported derived_features"):
            load_ohlcv_csv(
                str(self.csv_path),
                derived_features=["future_magic"],
            )

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

    def test_close_return_log_return_and_volatility_sliding_windows(self) -> None:
        """Build correctly shaped close, return, log-return, and volatility targets."""
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
        log_return_x, log_return_y, log_return_dates = build_sliding_windows(
            data,
            sequence_length=10,
            target_type="log_return",
        )
        volatility_x, volatility_y, volatility_dates = build_sliding_windows(
            data,
            sequence_length=10,
            target_type="volatility_5",
        )

        self.assertEqual(close_x.shape, (110, 10, 5))
        self.assertEqual(close_y.shape, (110, 1))
        self.assertEqual(return_x.shape, (110, 10, 5))
        self.assertEqual(return_y.shape, (110, 1))
        self.assertEqual(log_return_x.shape, (110, 10, 5))
        self.assertEqual(log_return_y.shape, (110, 1))
        self.assertEqual(volatility_x.shape, (106, 10, 5))
        self.assertEqual(volatility_y.shape, (106, 1))
        self.assertEqual(close_dates, return_dates)
        self.assertEqual(close_dates, log_return_dates)
        self.assertEqual(close_dates[:106], volatility_dates)
        self.assertAlmostEqual(float(close_y[0, 0]), 110.0)
        self.assertAlmostEqual(float(return_y[0, 0]), 110.0 / 109.0 - 1.0)
        self.assertAlmostEqual(float(log_return_y[0, 0]), np.log(110.0 / 109.0))
        expected_volatility = np.std(
            np.log(
                np.array([110.0, 111.0, 112.0, 113.0, 114.0])
                / np.array([109.0, 110.0, 111.0, 112.0, 113.0])
            ),
            ddof=0,
        )
        self.assertAlmostEqual(float(volatility_y[0, 0]), expected_volatility)
        self.assertTrue(np.all(volatility_y >= 0.0))

    def test_log_return_rejects_non_positive_close(self) -> None:
        """Reject log-return targets when either close value is non-positive."""
        data = load_ohlcv_csv(str(self.csv_path))
        data.loc[9, "Close"] = 0.0
        with self.assertRaisesRegex(ValueError, "Close values must be positive"):
            build_sliding_windows(
                data,
                sequence_length=10,
                target_type="log_return",
            )

    def test_volatility_rejects_non_positive_close(self) -> None:
        """Reject volatility targets when any close value is non-positive."""
        data = load_ohlcv_csv(str(self.csv_path))
        data.loc[0, "Close"] = -1.0
        with self.assertRaisesRegex(ValueError, "Close values must be positive"):
            build_sliding_windows(
                data,
                sequence_length=10,
                target_type="volatility_5",
            )

    def test_create_dataloaders_batch_shapes_and_targets(self) -> None:
        """Create chronological close, return, log-return, and volatility loaders."""
        close_bundle = create_dataloaders(
            str(self.csv_path),
            sequence_length=10,
            batch_size=8,
            target_type="close",
        )
        batch_x, batch_y = next(iter(close_bundle.train_loader))

        self.assertEqual(tuple(batch_x.shape), (8, 10, 5))
        self.assertEqual(tuple(batch_y.shape), (8, 1))
        self.assertIsNotNone(close_bundle.target_scaler)
        self.assertTrue(close_bundle.preprocessing_config["target_scaler_enabled"])
        self.assertEqual(
            close_bundle.preprocessing_config["target_scaler"],
            "StandardScaler",
        )
        restored_first_target = close_bundle.target_scaler.inverse_transform(
            close_bundle.train_dataset.targets[0].numpy().reshape(1, 1)
        )[0, 0]
        self.assertAlmostEqual(float(restored_first_target), 110.0, places=5)
        self.assertLess(
            abs(float(close_bundle.train_dataset.targets.mean())),
            0.5,
        )
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
        self.assertIsNone(return_bundle.target_scaler)
        self.assertFalse(return_bundle.preprocessing_config["target_scaler_enabled"])
        self.assertTrue(torch.isfinite(return_y).all())

        log_return_bundle = create_dataloaders(
            str(self.csv_path),
            sequence_length=10,
            batch_size=8,
            target_type="log_return",
        )
        log_return_x, log_return_y = next(iter(log_return_bundle.train_loader))
        self.assertEqual(tuple(log_return_x.shape), (8, 10, 5))
        self.assertEqual(tuple(log_return_y.shape), (8, 1))
        self.assertIsNone(log_return_bundle.target_scaler)
        self.assertFalse(
            log_return_bundle.preprocessing_config["target_scaler_enabled"]
        )
        self.assertEqual(
            log_return_bundle.preprocessing_config["target_type"],
            "log_return",
        )
        self.assertTrue(torch.isfinite(log_return_y).all())

        volatility_bundle = create_dataloaders(
            str(self.csv_path),
            sequence_length=10,
            batch_size=4,
            target_type="volatility_5",
        )
        volatility_x, volatility_y = next(iter(volatility_bundle.train_loader))
        self.assertEqual(tuple(volatility_x.shape), (4, 10, 5))
        self.assertEqual(tuple(volatility_y.shape), (4, 1))
        self.assertIsNone(volatility_bundle.target_scaler)
        self.assertFalse(
            volatility_bundle.preprocessing_config["target_scaler_enabled"]
        )
        self.assertEqual(
            volatility_bundle.preprocessing_config["target_type"],
            "volatility_5",
        )
        self.assertTrue(torch.isfinite(volatility_y).all())
        self.assertTrue((volatility_y >= 0.0).all())
        self.assertEqual(
            (
                len(volatility_bundle.train_dataset),
                len(volatility_bundle.val_dataset),
                len(volatility_bundle.test_dataset),
            ),
            (70, 4, 4),
        )

        derived_features = [
            "log_return",
            "abs_log_return",
            "high_low_range",
            "close_open_return",
            "rolling_vol_5",
            "rolling_vol_10",
            "rolling_vol_20",
            "volume_change",
        ]
        feature_columns = DEFAULT_FEATURE_COLUMNS + derived_features
        feature_bundle = create_dataloaders(
            str(self.csv_path),
            sequence_length=10,
            batch_size=4,
            feature_columns=feature_columns,
            derived_features=derived_features,
            target_type="volatility_5",
        )
        feature_x, feature_y = next(iter(feature_bundle.train_loader))
        self.assertEqual(tuple(feature_x.shape), (4, 10, 13))
        self.assertEqual(tuple(feature_y.shape), (4, 1))
        self.assertEqual(
            feature_bundle.preprocessing_config["derived_features"],
            derived_features,
        )
        self.assertEqual(
            feature_bundle.preprocessing_config["feature_columns"],
            feature_columns,
        )


if __name__ == "__main__":
    unittest.main()
