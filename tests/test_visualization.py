"""Tests for publication-oriented experiment visualizations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.utils.visualization import (
    plot_error_histogram,
    plot_lct_parameter_history,
    plot_loss_curves,
    plot_prediction_curve,
    plot_residual_distribution,
)


class VisualizationTest(unittest.TestCase):
    """Verify each plotting function creates non-empty PNG and PDF files."""

    def setUp(self) -> None:
        """Create an isolated figure directory and deterministic sample values."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.figures_dir = Path(self.temp_dir.name) / "中文路径" / "figures"
        self.y_true = np.linspace(0.0, 1.0, 40)
        self.y_pred = self.y_true + 0.05 * np.sin(np.arange(40))

    def tearDown(self) -> None:
        """Remove all temporary image files."""
        self.temp_dir.cleanup()

    def test_loss_curves_create_png_and_pdf(self) -> None:
        """Save train and validation losses in both required formats."""
        paths = plot_loss_curves(
            torch.tensor([1.0, 0.8, 0.6]),
            torch.tensor([1.1, 0.9, 0.7]),
            self.figures_dir,
        )
        self._assert_figure_pair(paths, "loss_curve")

    def test_prediction_curve_creates_png_and_pdf(self) -> None:
        """Save prediction and ground-truth curves."""
        paths = plot_prediction_curve(
            self.y_true,
            self.y_pred,
            self.figures_dir,
        )
        self._assert_figure_pair(paths, "prediction_curve")

    def test_residual_distribution_creates_png_and_pdf(self) -> None:
        """Save residual values across the time index."""
        paths = plot_residual_distribution(
            self.y_true,
            self.y_pred,
            self.figures_dir,
        )
        self._assert_figure_pair(paths, "residual_distribution")

    def test_error_histogram_creates_png_and_pdf(self) -> None:
        """Save the prediction-error histogram."""
        paths = plot_error_histogram(
            self.y_true,
            self.y_pred,
            self.figures_dir,
        )
        self._assert_figure_pair(paths, "error_histogram")

    def test_lct_parameter_history_creates_png_and_pdf(self) -> None:
        """Save learned LCT-Riesz parameter trajectories."""
        history = [
            {
                "epoch": 1,
                "alpha": 1.0,
                "m": 1.0,
                "q": 0.0,
                "gamma": 1.0,
            },
            {
                "epoch": 2,
                "alpha": 0.99,
                "m": 1.01,
                "q": 0.02,
                "gamma": 1.0,
            },
        ]
        paths = plot_lct_parameter_history(history, self.figures_dir)
        self._assert_figure_pair(paths, "lct_parameter_analysis")

    def _assert_figure_pair(
        self,
        paths: tuple[Path, Path],
        expected_stem: str,
    ) -> None:
        """Assert the returned PNG and PDF paths exist and are non-empty."""
        png_path, pdf_path = paths
        self.assertEqual(png_path.name, "{}.png".format(expected_stem))
        self.assertEqual(pdf_path.name, "{}.pdf".format(expected_stem))
        for path in paths:
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
