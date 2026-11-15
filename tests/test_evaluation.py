"""Test the evaluation of detected R-peaks."""

import math
import unittest

import numpy as np

from rpeak_detection.evaluation import (EvaluationResult,
                                        select_beat_positions,
                                        evaluate_detection)


class TestEvaluationResult(unittest.TestCase):
    """Tests for EvaluationResult."""

    def test_compute_sensitivity(self):
        """Return two thirds for two true positives and one false negative."""
        result = EvaluationResult(2, 1, 1)
        np.testing.assert_allclose(result.compute_sensitivity(), 2 / 3)

    def test_compute_positive_predictive_value(self):
        """Return two thirds for two true positives and one false positive."""
        result = EvaluationResult(2, 1, 1)
        np.testing.assert_allclose(result.compute_positive_predictive_value(),
                                   2 / 3)

    def test_compute_f1_score(self):
        """Return two thirds for one false positive and one false negative."""
        result = EvaluationResult(2, 1, 1)
        np.testing.assert_allclose(result.compute_f1_score(), 2 / 3)

    def test_return_nan_sensitivity_without_beats(self):
        """Return NaN for the sensitivity if no beats are annotated."""
        result = EvaluationResult(0, 0, 0)
        self.assertTrue(math.isnan(result.compute_sensitivity()))

    def test_return_zero_f1_without_true_positives(self):
        """Return zero for the F1 score if no beat is detected correctly."""
        result = EvaluationResult(0, 3, 2)
        self.assertEqual(result.compute_f1_score(), 0.0)


class TestSelectBeatPositions(unittest.TestCase):
    """Tests for select_beat_positions."""

    def test_keep_only_beat_symbols(self):
        """Return only the positions of annotations that represent beats."""
        annotated_positions = np.array([18, 77, 370, 662])
        annotated_symbols = ["+", "N", "A", "~"]
        beat_positions = select_beat_positions(annotated_positions,
                                               annotated_symbols)
        np.testing.assert_array_equal(beat_positions, np.array([77, 370]))


class TestEvaluateDetection(unittest.TestCase):
    """Tests for evaluate_detection."""

    def test_match_nearest_detection_once(self):
        """Count each detection once and prefer the first one in a tie."""
        detected_positions = np.array([9, 11, 49, 70])
        annotated_positions = np.array([10, 50, 90])
        annotated_symbols = ["N", "N", "N"]
        result = evaluate_detection(detected_positions, annotated_positions,
                                    annotated_symbols,
                                    sampling_rate=1000, tolerance_ms=3)
        self.assertEqual(result.true_positives, 2)
        self.assertEqual(result.false_positives, 2)
        self.assertEqual(result.false_negatives, 1)
