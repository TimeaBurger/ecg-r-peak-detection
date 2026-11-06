"""Test the preprocessing functions."""

import unittest

import numpy as np

from rpeak_detection.preprocessing import (compute_derivative, square_signal,
                                           apply_moving_window_integration,
                                           apply_low_pass_filter,
                                           apply_high_pass_filter,
                                           resample_signal)


class TestComputeDerivative(unittest.TestCase):
    """Test the compute_derivative function."""

    def test_ramp_has_slope_one(self):
        """Return a slope of one for a linear ramp."""
        result = compute_derivative(np.arange(10))
        np.testing.assert_allclose(result[4], 1)

    def test_impulse_response_is_antisymmetric(self):
        """Return an antisymmetric impulse response."""
        impulse = np.zeros(9)
        impulse[4] = 1
        result = compute_derivative(impulse)
        expected = np.array([0, 0, 0.125, 0.25, 0, -0.25, -0.125, 0, 0])
        np.testing.assert_allclose(result, expected)


class TestSquareSignal(unittest.TestCase):
    """Test the square_signal function."""

    def test_square_each_value(self):
        """Return the square of each value, also for negative values."""
        result = square_signal(np.array([-3, -1, 0, 2, 10]))
        expected = np.array([9, 1, 0, 4, 100])
        np.testing.assert_allclose(result, expected)


class TestApplyMovingWindowIntegration(unittest.TestCase):
    """Test the apply_moving_window_integration function."""

    def test_window_fills_up_after_thirty_samples(self):
        """Reach the full average after thirty samples."""
        result = apply_moving_window_integration(np.ones(40))
        np.testing.assert_allclose(result[9], 1 / 3)
        np.testing.assert_allclose(result[29], 1)
        np.testing.assert_allclose(result[35], 1)


class TestApplyLowPassFilter(unittest.TestCase):
    """Test the apply_low_pass_filter function."""

    def test_impulse_response_is_triangle(self):
        """Return a triangular impulse response with a gain of 36."""
        impulse = np.zeros(15)
        impulse[0] = 1
        result = apply_low_pass_filter(impulse)
        expected = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0])
        np.testing.assert_allclose(result, expected)


class TestApplyHighPassFilter(unittest.TestCase):
    """Test the apply_high_pass_filter function."""

    def test_impulse_response_sums_to_zero(self):
        """Return an impulse response peaking at 16 and summing to 0."""
        impulse = np.zeros(40)
        impulse[0] = 1
        result = apply_high_pass_filter(impulse)
        self.assertEqual(np.argmax(result), 16)
        np.testing.assert_allclose(np.sum(result), 0, atol=1e-12)


class TestResampleSignal(unittest.TestCase):
    """Test the resample_signal function."""

    def test_ten_seconds_give_2000_samples(self):
        """Return 2000 samples for ten seconds of signal."""
        result = resample_signal(np.zeros(3600), 360)
        self.assertEqual(len(result), 2000)


if __name__ == "__main__":
    unittest.main()
