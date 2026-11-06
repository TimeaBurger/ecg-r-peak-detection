"""Detect R-peaks in ECG signals.

Provide a base class for R-peak detectors and two implementations:
a simple threshold-based baseline and the Pan-Tompkins algorithm.
"""

import numpy as np

from rpeak_detection.preprocessing import (resample_signal,
                                           apply_low_pass_filter,
                                           apply_high_pass_filter,
                                           compute_derivative, square_signal,
                                           apply_moving_window_integration,
                                           TARGET_SAMPLING_RATE)

REFRACTORY_PERIOD_MS = 200  # refractory period from Pan and Tompkins (1985)
LEARNING_PHASE_MS = 2000  # learning phase 1 from Pan and Tompkins (1985)


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


def locate_candidate_peaks(signal, search_radius):
    """Locate local maxima that are highest within the search radius."""
    candidate_positions = []
    for n in range(1, len(signal) - 1):
        if signal[n - 1] < signal[n] and signal[n] > signal[n + 1]:
            if n - search_radius >= 0:
                window_start = n - search_radius
            else:
                window_start = 0
            window_end = n + search_radius + 1
            if signal[n] == np.max(signal[window_start:window_end]):
                candidate_positions.append(n)
    return np.array(candidate_positions)


def select_signal_peaks(signal, candidate_positions, learning_samples):
    """Select signal peaks from candidates using an adaptive threshold."""
    learning_phase = signal[:learning_samples]
    signal_peak_estimate = np.max(learning_phase)
    noise_peak_estimate = np.mean(learning_phase)
    threshold = noise_peak_estimate + 0.25 * (signal_peak_estimate
                                              - noise_peak_estimate)
    signal_peak_positions = []
    for position in candidate_positions:
        peak_value = signal[position]
        if peak_value > threshold:
            signal_peak_positions.append(position)
            signal_peak_estimate = (0.125 * peak_value
                                    + 0.875 * signal_peak_estimate)
        else:
            noise_peak_estimate = (0.125 * peak_value
                                   + 0.875 * noise_peak_estimate)
        threshold = noise_peak_estimate + 0.25 * (signal_peak_estimate
                                                  - noise_peak_estimate)
    return np.array(signal_peak_positions)


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


class PanTompkinsDetector(RPeakDetector):
    """Represent an R-peak detector based on the Pan-Tompkins algorithm."""

    def __init__(self, threshold_ratio=0.6):
        self.threshold_ratio = threshold_ratio

    def detect_r_peaks(self, signal, sampling_rate):
        """."""
        resampled_signal = resample_signal(signal, sampling_rate)
        low_passed_signal = apply_low_pass_filter(resampled_signal)
        high_passed_signal = apply_high_pass_filter(low_passed_signal)
        derivative_signal = compute_derivative(high_passed_signal)
        squared_signal = square_signal(derivative_signal)
        integrated_signal = apply_moving_window_integration(squared_signal)
        threshold = self.threshold_ratio * np.max(integrated_signal)
        peak_positions = locate_peaks_above_threshold(integrated_signal,
                                                      threshold)
        scale_factor = sampling_rate / TARGET_SAMPLING_RATE
        r_peak_positions = np.round(peak_positions * scale_factor).astype(int)
        return r_peak_positions
