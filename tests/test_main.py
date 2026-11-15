"""Test the functions of the main module."""

import unittest

import numpy as np
import pandas as pd

from main import (
    compute_hrv_metrics, select_record_names, compute_overall_metrics
)


class TestSelectRecordNames(unittest.TestCase):
    """Tests for select_record_names."""

    def test_remove_excluded_record_names(self):
        """Return only the record names that are not excluded."""
        self.assertEqual(
            select_record_names(["100", "101", "102", "103"], ["102"]),
            ["100", "101", "103"]
        )


class TestComputeHrvMetrics(unittest.TestCase):
    """Tests for compute_hrv_metrics."""

    def test_return_sdnn_and_rmssd(self):
        """Return 25 and 50 ms for intervals of 1000 and 1050 ms."""
        metrics = compute_hrv_metrics(np.array([0, 360, 738]), 360)
        np.testing.assert_allclose(metrics["sdnn"], 25)
        np.testing.assert_allclose(metrics["rmssd"], 50)


class TestComputeOverallMetrics(unittest.TestCase):
    """Tests for compute_overall_metrics."""

    def test_add_counts_before_computing_sensitivity(self):
        """Compute the sensitivity from the summed counts of all records."""
        results = pd.DataFrame([
            {
                "detector_name": "baseline",
                "true_positives": 90,
                "false_positives": 0,
                "false_negatives": 10,
            },
            {
                "detector_name": "baseline",
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 10,
            },
        ])
        overall_metrics = compute_overall_metrics(results)
        np.testing.assert_allclose(overall_metrics["sensitivity"][0], 90 / 110)
