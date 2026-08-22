"""Tests for no-training residual contribution diagnostics."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from diagnose_residual_contribution import (
    build_constant_control_predictions,
    circular_shift_permutations,
    permutation_metric_rows,
    random_permutations,
    summarize_permutations,
)


class ResidualContributionDiagnosticTest(unittest.TestCase):
    """Validate deterministic permutations and label-safe bias controls."""

    def test_random_permutations_are_reproducible_and_preserve_values(self) -> None:
        """Repeat a seeded permutation sequence without changing its values."""
        correction = np.asarray([-0.4, -0.3, -0.2, -0.1], dtype=np.float64)
        first = random_permutations(correction, permutation_count=8, master_seed=42)
        second = random_permutations(correction, permutation_count=8, master_seed=42)

        self.assertEqual(len(first), 8)
        for first_item, second_item in zip(first, second):
            self.assertEqual(first_item[0], second_item[0])
            self.assertEqual(first_item[1], second_item[1])
            np.testing.assert_array_equal(first_item[2], second_item[2])
            np.testing.assert_array_equal(
                np.sort(first_item[2]),
                np.sort(correction),
            )

    def test_circular_shifts_exclude_original_alignment_and_preserve_values(self) -> None:
        """Generate every nonzero shift without changing correction values."""
        correction = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        shifted = circular_shift_permutations(correction)

        self.assertEqual([item[1] for item in shifted], [1, 2, 3])
        for _, shift, values in shifted:
            self.assertNotEqual(shift, 0)
            self.assertFalse(np.array_equal(values, correction))
            np.testing.assert_array_equal(np.sort(values), np.sort(correction))

    def test_constant_controls_use_only_allowed_sources(self) -> None:
        """Keep correction means label-free and fit biases on validation only."""
        test_components = {
            "target": np.asarray([1000.0, -1000.0]),
            "main_pred": np.asarray([1.0, 2.0]),
            "residual_correction": np.asarray([-0.4, -0.2]),
        }
        validation_components = {
            "target": np.asarray([2.0, 5.0, 7.0]),
            "main_pred": np.asarray([1.0, 2.0, 3.0]),
            "residual_correction": np.asarray([-0.3, -0.6, -0.9]),
        }
        predictions, offsets = build_constant_control_predictions(
            test_components,
            validation_components,
        )

        self.assertAlmostEqual(offsets["test_correction_mean"], -0.3)
        self.assertAlmostEqual(offsets["validation_correction_mean"], -0.6)
        self.assertAlmostEqual(offsets["validation_mse_bias"], 8.0 / 3.0)
        self.assertAlmostEqual(offsets["validation_mae_bias"], 3.0)
        np.testing.assert_allclose(
            predictions["constant_test_correction_mean"],
            np.asarray([0.7, 1.7]),
        )
        np.testing.assert_allclose(
            predictions["constant_validation_correction_mean"],
            np.asarray([0.4, 1.4]),
        )

        changed_test_targets = dict(test_components)
        changed_test_targets["target"] = np.asarray([-1e9, 1e9])
        repeated_predictions, repeated_offsets = build_constant_control_predictions(
            changed_test_targets,
            validation_components,
        )
        self.assertEqual(offsets, repeated_offsets)
        for name in predictions:
            np.testing.assert_array_equal(
                predictions[name],
                repeated_predictions[name],
            )

    def test_permutation_metrics_are_finite_and_p_value_direction_is_correct(self) -> None:
        """Use lower error and higher R² directions for empirical p values."""
        target = np.asarray([0.0, 1.0, 2.0, 3.0])
        main = np.asarray([0.1, 0.8, 2.2, 2.7])
        correction = np.asarray([-0.1, 0.2, -0.2, 0.3])
        generated = permutation_metric_rows(
            target,
            main,
            correction,
            random_count=5,
            master_seed=42,
        )
        self.assertEqual(len(generated), 8)
        for row in generated:
            for metric_name in ("mae", "mse", "rmse", "r2"):
                self.assertTrue(np.isfinite(row[metric_name]))

        rows = []
        for permutation_type in ("random", "circular_shift"):
            for index, value in enumerate((0.5, 1.0, 2.0), start=1):
                rows.append(
                    {
                        "permutation_type": permutation_type,
                        "permutation_id": index,
                        "shift": index if permutation_type == "circular_shift" else None,
                        "seed": index if permutation_type == "random" else None,
                        "mae": value,
                        "mse": value,
                        "rmse": value,
                        "r2": (0.4, 0.5, 0.6)[index - 1],
                    }
                )
        summary = summarize_permutations(
            rows,
            {"mae": 1.0, "mse": 1.0, "rmse": 1.0, "r2": 0.5},
        )
        random_metrics = summary["random"]["metrics"]
        self.assertAlmostEqual(random_metrics["mae"]["empirical_p_value"], 0.75)
        self.assertAlmostEqual(random_metrics["r2"]["empirical_p_value"], 0.75)
        self.assertAlmostEqual(
            random_metrics["mae"]["full_better_than_proportion"],
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            random_metrics["r2"]["full_better_than_proportion"],
            1.0 / 3.0,
        )

    def test_existing_full_diagnostic_matches_historical_experiment(self) -> None:
        """Require the local formal Full verification to remain exact."""
        project_root = Path(__file__).resolve().parents[1]
        metrics_path = (
            project_root
            / "experiments"
            / "lstm"
            / "diagnostics"
            / "residual_contribution"
            / "20260625_163554"
            / "residual_ablation_metrics.json"
        )
        if not metrics_path.is_file():
            self.skipTest("Local formal diagnostic output is unavailable.")
        document = json.loads(metrics_path.read_text(encoding="utf-8"))
        verification = document["verification"]
        self.assertTrue(verification["matched"])
        for difference in verification["metric_absolute_differences"].values():
            self.assertLessEqual(float(difference), 1e-8)


if __name__ == "__main__":
    unittest.main()
