"""Detect R-peaks in ECG signals.

Provide a base class for R-peak detectors and two implementations:
a simple threshold-based baseline and the Pan-Tompkins algorithm.
"""

import numpy as np


def locate_peaks_above_threshold(signal, threshold):
    """Locate the highest sample of each region above the threshold."""
    above_threshold = np.array(signal) > threshold
    r_peak_positions = []
    for n in range(1, len(above_threshold)):
        if above_threshold[n] and not above_threshold[n - 1]:
            peak_start = n
        if not above_threshold[n] and above_threshold[n - 1]:
            peak_end = n
            r_peak_position = (
                np.argmax(signal[peak_start:peak_end]) + peak_start
            )
            r_peak_positions.append(r_peak_position)
    return np.array(r_peak_positions)


class RPeakDetector:
    """Represent a generic R-peak detector."""

    def detect_r_peaks(self, signal, sampling_rate):
        """Detect the R-peaks in the given ECG signal.

        Return an array containing the sample positions of the detected
        R-peaks.
        Raise NotImplementedError if subclasses do not implement this method.
        """
        raise NotImplementedError(
            "Subclasses must implement the detect_r_peaks method."
        )


class ThresholdBasedDetector(RPeakDetector):
    """Represent an R-peak detector based on a fixed amplitude threshold."""

    def __init__(self, threshold_ratio=0.6):
        self.threshold_ratio = threshold_ratio

    def detect_r_peaks(self, signal, sampling_rate):
        """Detect R-peaks using a threshold relative to the signal maximum."""
        threshold = self.threshold_ratio * np.max(signal)
        return locate_peaks_above_threshold(signal, threshold)
