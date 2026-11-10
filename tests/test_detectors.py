"""Test the R-peak detectors."""

import unittest

import numpy as np

from rpeak_detection.detectors import (RPeakDetector, ThresholdBasedDetector,
                                      locate_candidate_peaks,
                                      select_signal_peaks,
                                      refine_peak_positions)


class TestRPeakDetector(unittest.TestCase):

    def test_raise_not_implemented_error(self):
        """Raise NotImplementedError when the base class is used."""
        detector = RPeakDetector()
        with self.assertRaises(NotImplementedError):
            detector.detect_r_peaks(np.array([0, 1, 0]), 360)


class TestThresholdBasedDetector(unittest.TestCase):

    def test_return_one_peak_per_region(self):
        """Return the highest sample of each region above the threshold."""
        signal = np.array([0, 1, 2, 7, 10, 8, 1, 0, 1, 3, 7, 9, 8, 2, 0])
        detector = ThresholdBasedDetector()
        r_peak_positions = detector.detect_r_peaks(signal, 360)
        np.testing.assert_array_equal(r_peak_positions, np.array([4, 11]))


class TestLocateCandidatePeaks(unittest.TestCase):

    def test_keep_highest_maximum_within_radius(self):
        """Return only the highest local maximum within the search radius."""
        signal = np.array([0, 1, 0, 3, 2, 5, 0, 0, 4, 0])
        candidate_positions = locate_candidate_peaks(signal, 2)
        np.testing.assert_array_equal(candidate_positions, np.array([5, 8]))


class TestSelectSignalPeaks(unittest.TestCase):

    def test_select_candidates_above_adaptive_threshold(self):
        """Return only the candidates above the adaptive threshold."""
        signal = np.array([0, 8, 0, 0, 2, 0, 9, 0, 1, 0, 7])
        candidate_positions = np.array([4, 6, 8, 10])
        signal_peak_positions = select_signal_peaks(signal,
                                                    candidate_positions, 4)
        np.testing.assert_array_equal(signal_peak_positions, np.array([6, 10]))


class TestRefinePeakPositions(unittest.TestCase):

    def test_move_to_highest_sample(self):
        """Move each position to the highest sample within the radius."""
        np.testing.assert_array_equal(
            refine_peak_positions(
                np.array([0, 1, 5, 2, 0, 0, 3, 9, 4, 0]),
                np.array([3, 6]), 2), np.array([2, 7])
        )

    def test_clip_window_at_signal_start(self):
        """Start the search window at zero near the signal start."""
        np.testing.assert_array_equal(
            refine_peak_positions(
                np.array([0, 1, 5, 2, 0, 0, 3, 9, 4, 0]),
                np.array([1]), 2), np.array([2])
        )
