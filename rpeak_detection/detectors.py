"""Detect R-peaks in ECG signals.

Provide a base class for R-peak detectors and two implementations:
a simple threshold-based baseline and the Pan-Tompkins algorithm.
"""


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
