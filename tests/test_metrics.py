"""Tests for financial forecasting regression metrics."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.utils.metrics import (
    calculate_all_metrics,
    directional_accuracy,
    mae,
    mape,
    mse,
    r2_score,
    rmse,
)


class MetricsTest(unittest.TestCase):
    """Verify numerical correctness and supported input types."""

    def test_numpy_regression_metrics_are_correct(self) -> None:
        """Calculate known MAE, MSE, RMSE, MAPE, and R-squared values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])

        self.assertAlmostEqual(mae(y_true, y_pred), 1.0 / 3.0)
        self.assertAlmostEqual(mse(y_true, y_pred), 1.0 / 6.0)
        self.assertAlmostEqual(rmse(y_true, y_pred), np.sqrt(1.0 / 6.0))
        self.assertAlmostEqual(mape(y_true, y_pred), 200.0 / 9.0)
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.75)

    def test_torch_directional_accuracy_and_complete_metrics(self) -> None:
        """Support tensor inputs and compare positive, negative, and zero signs."""
        y_true = torch.tensor([0.1, -0.2, 0.0, 0.3])
        y_pred = torch.tensor([0.2, 0.1, 0.0, -0.4])

        self.assertAlmostEqual(directional_accuracy(y_true, y_pred), 0.5)
        metrics = calculate_all_metrics(y_true, y_pred)
        self.assertEqual(
            set(metrics),
            {
                "mae",
                "mse",
                "rmse",
                "mape",
                "r2",
                "directional_accuracy",
            },
        )
        self.assertTrue(all(isinstance(value, float) for value in metrics.values()))

    def test_mape_and_r2_edge_cases_are_finite(self) -> None:
        """Avoid division by zero and undefined constant-target R-squared."""
        self.assertTrue(np.isfinite(mape([0.0, 1.0], [0.1, 1.0])))
        self.assertEqual(r2_score([2.0, 2.0], [2.0, 2.0]), 1.0)
        self.assertEqual(r2_score([2.0, 2.0], [1.0, 3.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
