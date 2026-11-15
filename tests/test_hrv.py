"""Test the heart rate variability (HRV) functions."""

import unittest

import numpy as np

from rpeak_detection.hrv import (compute_rr_intervals,
                                 compute_sdnn,
                                 compute_rmssd)


class TestComputeRRIntervals(unittest.TestCase):
    """Tests for compute_rr_intervals."""

    def test_convert_distances_to_milliseconds(self):
        """Return 1000 and 1050 ms for distances of 360 and 378 samples."""
        np.testing.assert_allclose(
            compute_rr_intervals(np.array([100, 460, 838]), 360), [1000, 1050]
        )


class TestComputeSdnn(unittest.TestCase):
    """Tests for compute_sdnn."""

    def test_return_standard_deviation(self):
        """Return the square root of 250 for deviations of 20 and 10 ms."""
        np.testing.assert_allclose(
            compute_sdnn(np.array([780, 820, 790, 810])), np.sqrt(250)
        )

    def test_return_zero_for_constant_intervals(self):
        """Return zero if all intervals are equal."""
        self.assertEqual(
            compute_sdnn(np.array([800, 800, 800, 800])), 0.0
        )


class TestComputeRmssd(unittest.TestCase):
    """Tests for compute_rmssd."""

    def test_return_root_mean_square_of_differences(self):
        """Return sqrt(2900 / 3) for differences of 40, -30 and 20 ms."""
        np.testing.assert_allclose(
            compute_rmssd(np.array([780, 820, 790, 810])), np.sqrt(2900 / 3)
        )

    def test_return_zero_for_constant_intervals(self):
        """Return zero if all intervals are equal."""
        self.assertEqual(
            compute_rmssd(np.array([800, 800, 800, 800])), 0.0
        )
