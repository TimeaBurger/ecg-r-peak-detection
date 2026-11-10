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
PROCESSING_DELAY = 35.5  # samples at 200 Hz: 5 + 16 + 14.5 (impulse tests)
RR_AVERAGE_LENGTH = 8  # RR AVERAGE2 length, Pan and Tompkins (1985), Eq. 25
RR_LOW_LIMIT = 0.92  # RR LOW LIMIT, Pan and Tompkins (1985), Eq. 26
RR_HIGH_LIMIT = 1.16  # RR HIGH LIMIT, Pan and Tompkins (1985), Eq. 27
RR_MISSED_LIMIT = 1.66  # RR MISSED LIMIT, Pan and Tompkins (1985), Eq. 28
R_PEAK_SEARCH_RADIUS_MS = 50  # half of max. normal QRS, Dupre et al. (2005)


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
    rr_intervals = []
    rr_average = None
    for position in candidate_positions:
        peak_value = signal[position]
        if peak_value > threshold:
            if len(signal_peak_positions) > 0:
                rr_interval = position - signal_peak_positions[-1]
                if (rr_average is not None
                        and rr_interval > RR_MISSED_LIMIT * rr_average):
                    missed_position = search_back(
                        signal, candidate_positions, signal_peak_positions[-1],
                        position, threshold)
                    if missed_position is not None:
                        signal_peak_positions.append(missed_position)
                        signal_peak_estimate = (0.25 * signal[missed_position]
                                                + 0.75 * signal_peak_estimate)
                        rr_interval = position - signal_peak_positions[-1]
                if (rr_average is None
                        or RR_LOW_LIMIT * rr_average <= rr_interval
                        <= RR_HIGH_LIMIT * rr_average):
                    rr_intervals.append(rr_interval)
                    rr_average = np.mean(rr_intervals[-RR_AVERAGE_LENGTH:])
            signal_peak_positions.append(position)
            signal_peak_estimate = (0.125 * peak_value
                                    + 0.875 * signal_peak_estimate)
        else:
            noise_peak_estimate = (0.125 * peak_value
                                   + 0.875 * noise_peak_estimate)
        threshold = noise_peak_estimate + 0.25 * (signal_peak_estimate
                                                  - noise_peak_estimate)
    return np.array(signal_peak_positions)

def search_back(signal, candidate_positions, previous_peak_position,
                current_peak_position, threshold):
    """Return the highest candidate between two peaks above half the threshold.

    Return None if no candidate exceeds half the threshold.
    """
    search_threshold = 0.5 * threshold
    best_position = None
    best_value = search_threshold
    for position in candidate_positions:
        if (previous_peak_position < position < current_peak_position
                and signal[position] > best_value):
            best_position = position
            best_value = signal[position]
    return best_position


def refine_peak_positions(signal, peak_positions, search_radius):
    """Move each peak position to the highest sample within the radius."""
    refined_positions = []
    for position in peak_positions:
        if position - search_radius >= 0:
            window_start = position - search_radius
        else:
            window_start = 0
        window_end = position + search_radius + 1
        peak_position = (
            np.argmax(signal[window_start:window_end]) + window_start
        )
        refined_positions.append(peak_position)
    return np.array(refined_positions)


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

    def detect_r_peaks(self, signal, sampling_rate):
        """Detect R-peaks using the Pan-Tompkins algorithm."""
        resampled_signal = resample_signal(signal, sampling_rate)
        low_passed_signal = apply_low_pass_filter(resampled_signal)
        high_passed_signal = apply_high_pass_filter(low_passed_signal)
        derivative_signal = compute_derivative(high_passed_signal)
        squared_signal = square_signal(derivative_signal)
        integrated_signal = apply_moving_window_integration(squared_signal)
        search_radius = round(TARGET_SAMPLING_RATE
                              * (REFRACTORY_PERIOD_MS / 1000))
        learning_samples = round(TARGET_SAMPLING_RATE
                                 * (LEARNING_PHASE_MS / 1000))
        candidate_positions = locate_candidate_peaks(integrated_signal,
                                                     search_radius)
        peak_positions = select_signal_peaks(integrated_signal,
                                             candidate_positions,
                                             learning_samples)
        scale_factor = sampling_rate / TARGET_SAMPLING_RATE
        corrected_positions = peak_positions - PROCESSING_DELAY
        r_peak_positions = np.round(corrected_positions
                                    * scale_factor).astype(int)
        r_peak_search_radius = round(sampling_rate
                                     * (R_PEAK_SEARCH_RADIUS_MS / 1000))
        return refine_peak_positions(signal,
                                     r_peak_positions, r_peak_search_radius)
