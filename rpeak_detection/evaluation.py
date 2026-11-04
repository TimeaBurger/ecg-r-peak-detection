"""Evaluate detected R-peak positions against expert annotations.

Match each annotation to the nearest detection within a tolerance window.
Count true positives, false positives and false negatives. Compute sensitivity,
positive predictive value and F1 score from them.
"""


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
