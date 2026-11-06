"""Test the preprocessing functions."""

import unittest

import numpy as np

from rpeak_detection.preprocessing import compute_derivative, square_signal


class TestComputeDerivative(unittest.TestCase):
    """Test the compute_derivative function."""

    def test_ramp_has_slope_one(self):
        """Return a slope of one for a linear ramp."""
        result = compute_derivative(np.arange(10))
        np.testing.assert_allclose(result[4], 1)

    def test_impulse_response_is_symmetric(self):
        """Return a symmetric impulse response."""
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


if __name__ == "__main__":
    unittest.main()
