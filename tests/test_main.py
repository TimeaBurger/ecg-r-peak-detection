"""Test the functions of the main module."""

import unittest

import numpy as np

from main import compute_hrv_metrics, select_record_names


class TestSelectRecordNames(unittest.TestCase):

    def test_remove_excluded_record_names(self):
        """Return only the record names that are not excluded."""
        self.assertEqual(
            select_record_names(["100", "101", "102", "103"], ["102"]),
            ["100", "101", "103"]
        )


class TestComputeHrvMetrics(unittest.TestCase):

    def test_return_sdnn_and_rmssd(self):
        """Return 25 and 50 ms for intervals of 1000 and 1050 ms."""
        metrics = compute_hrv_metrics(np.array([0, 360, 738]), 360)
        np.testing.assert_allclose(metrics["sdnn"], 25)
        np.testing.assert_allclose(metrics["rmssd"], 50)
