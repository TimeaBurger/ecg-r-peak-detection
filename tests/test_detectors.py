"""Test the R-peak detectors."""

import unittest

import numpy as np

from rpeak_detection.detectors import (RPeakDetector, ThresholdBasedDetector,
                                      locate_candidate_peaks,
                                      select_signal_peaks,
                                      refine_peak_positions, search_back,
                                      is_regular_rhythm)


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

    def test_skip_region_at_signal_start(self):
        """Skip a region that is above the threshold at the first sample."""
        signal = np.array([10, 8, 0, 0, 6, 0])
        detector = ThresholdBasedDetector()
        r_peak_positions = detector.detect_r_peaks(signal, 360)
        np.testing.assert_array_equal(r_peak_positions, np.array([4]))


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

    def test_find_missed_beat_with_search_back(self):
        """Find a small beat between two peaks with the search-back."""
        signal = np.zeros(60)
        signal[[10, 20, 30, 40, 50]] = [9, 9, 2, 9, 9]
        candidate_positions = np.array([10, 20, 30, 40, 50])
        np.testing.assert_array_equal(
            select_signal_peaks(signal, candidate_positions, 12),
            np.array([10, 20, 30, 40, 50])
        )

    def test_skip_beat_below_half_threshold(self):
        """Skip a beat below half the threshold even with the search-back."""
        signal = np.zeros(60)
        signal[[10, 20, 30, 40, 50]] = [9, 9, 2, 9, 9]
        signal[30] = 1
        candidate_positions = np.array([10, 20, 30, 40, 50])
        np.testing.assert_array_equal(
            select_signal_peaks(signal, candidate_positions, 12),
            np.array([10, 20, 40, 50])
        )


class TestSearchBack(unittest.TestCase):

    def test_return_candidate_above_half_threshold(self):
        """Return the candidate above half the threshold."""
        signal = np.zeros(60)
        signal[[10, 20, 30, 40, 50]] = [9, 9, 2, 9, 9]
        candidate_positions = np.array([10, 20, 30, 40, 50])
        self.assertEqual(search_back(signal, candidate_positions,
                                     20, 40, 3), 30)

    def test_return_none_below_half_threshold(self):
        """Return None if no candidate exceeds half the threshold."""
        signal = np.zeros(60)
        signal[[10, 20, 30, 40, 50]] = [9, 9, 2, 9, 9]
        candidate_positions = np.array([10, 20, 30, 40, 50])
        self.assertIsNone(search_back(signal, candidate_positions,
                                      20, 40, 8))


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


class TestIsRegularRhythm(unittest.TestCase):

    def test_return_true_for_regular_intervals(self):
        """Return True if all eight intervals are within the limits."""
        self.assertTrue(
            is_regular_rhythm([160, 160, 160, 160, 160, 160, 160, 160], 161)
        )

    def test_return_false_for_one_outlier(self):
        """Return False if one of the eight intervals is outside the limits."""
        self.assertFalse(
            is_regular_rhythm([160, 160, 160, 160, 160, 160, 160, 90], 161)
        )

    def test_return_true_for_fewer_than_eight_intervals(self):
        """Return True if there are fewer than eight intervals."""
        self.assertTrue(is_regular_rhythm([160, 90, 160], 161))
