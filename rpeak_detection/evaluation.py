"""Evaluate detected R-peak positions against expert annotations.

Match each annotation to the nearest detection within a tolerance window.
Count true positives, false positives and false negatives. Compute sensitivity,
positive predictive value and F1 score from them.
"""

import numpy as np

# Annotation symbols that represent beats
# as listed in the MIT-BIH Arrhythmia Database directory
BEAT_SYMBOLS = {"N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E",
                "/", "f", "Q"}


class EvaluationResult:
    """Represent the evaluation result for a single ECG recording."""

    def __init__(self, true_positives, false_positives, false_negatives):
        """Create a result from the given counts."""
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives

    def compute_sensitivity(self):
        """Return the sensitivity or NaN if no beats are annotated."""
        number_of_annotated_beats = self.true_positives + self.false_negatives
        if number_of_annotated_beats == 0:
            return float("nan")
        else:
            return self.true_positives / number_of_annotated_beats

    def compute_positive_predictive_value(self):
        """Return the positive predictive value.

        Return NaN if no beats are detected.
        """
        number_of_detected_beats = self.true_positives + self.false_positives
        if number_of_detected_beats == 0:
            return float("nan")
        else:
            return self.true_positives / number_of_detected_beats

    def compute_f1_score(self):
        """Return the F1 score.

        Return NaN if there are neither annotated nor detected beats.
        """
        denominator = (2 * self.true_positives + self.false_positives +
                       self.false_negatives)
        if denominator == 0:
            return float("nan")
        else:
            return 2 * self.true_positives / denominator


def select_beat_positions(annotated_positions, annotated_symbols):
    """Return the positions of the annotations that represent beats."""
    beat_positions = []
    for n in range(len(annotated_positions)):
        if annotated_symbols[n] in BEAT_SYMBOLS:
            beat_positions.append(annotated_positions[n])
    return np.array(beat_positions)


def evaluate_detection(detected_positions, annotated_positions,
                       annotated_symbols, sampling_rate, tolerance_ms=150):
    """Evaluate the detected R-peaks against the annotated beats.

    Return an EvaluationResult with the counts of true positives,
    false positives and false negatives.
    """
    beat_positions = select_beat_positions(annotated_positions,
                                           annotated_symbols)
    tolerance_samples = round(sampling_rate * (tolerance_ms / 1000))
    is_used = [False] * len(detected_positions)
    true_positives = 0
    false_negatives = 0
    for beat_position in beat_positions:
        best_index = None
        best_distance = tolerance_samples + 1
        for i in range(len(detected_positions)):
            distance = abs(beat_position - detected_positions[i])
            if not is_used[i] and distance <= tolerance_samples:
                if distance < best_distance:
                    best_index = i
                    best_distance = distance
        if best_index is None:
            false_negatives += 1
        else:
            is_used[best_index] = True
            true_positives += 1
    false_positives = is_used.count(False)
    return EvaluationResult(true_positives, false_positives, false_negatives)
