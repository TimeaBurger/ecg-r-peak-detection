"""Preprocess ECG signals for R-peak detection.

Resample the signals to 200 Hz and apply low-pass and high-pass filters,
as well as the derivative, squaring and moving-window integration stages
according to the Pan-Tompkins algorithm.
"""

import numpy as np
from scipy.signal import resample_poly

TARGET_SAMPLING_RATE = 200
WINDOW_SIZE = 30  # samples (150 ms at 200 Hz), Pan and Tompkins (1985)


def apply_low_pass_filter(signal):
    """Apply the low-pass filter to the signal."""
    filtered_signal = np.zeros(len(signal))
    for n in range(len(signal)):
        if n >= 1:
            output_1_before = filtered_signal[n - 1]
        else:
            output_1_before = 0
        if n >= 2:
            output_2_before = filtered_signal[n - 2]
        else:
            output_2_before = 0
        if n >= 6:
            input_6_before = signal[n - 6]
        else:
            input_6_before = 0
        if n >= 12:
            input_12_before = signal[n - 12]
        else:
            input_12_before = 0
        filtered_signal[n] = (signal[n] + 2 * output_1_before
                              - output_2_before - 2 * input_6_before
                              + input_12_before)
    return filtered_signal


def apply_high_pass_filter(signal):
    """Apply the high-pass filter to the signal."""
    filtered_signal = np.zeros(len(signal))
    for n in range(len(signal)):
        if n >= 1:
            output_1_before = filtered_signal[n - 1]
        else:
            output_1_before = 0
        if n >= 16:
            input_16_before = signal[n - 16]
        else:
            input_16_before = 0
        if n >= 17:
            input_17_before = signal[n - 17]
        else:
            input_17_before = 0
        if n >= 32:
            input_32_before = signal[n - 32]
        else:
            input_32_before = 0
        filtered_signal[n] = (output_1_before - signal[n] / 32
                              + input_16_before - input_17_before
                              + input_32_before / 32)
    return filtered_signal


def compute_derivative(signal):
    """Compute the derivative of the signal."""
    computed_derivative = np.zeros(len(signal))
    for n in range(len(signal)):
        if n >= 2:
            input_2_before = signal[n - 2]
        else:
            input_2_before = 0
        if n >= 1:
            input_1_before = signal[n - 1]
        else:
            input_1_before = 0
        if n + 1 <= len(signal) - 1:
            input_1_after = signal[n + 1]
        else:
            input_1_after = 0
        if n + 2 <= len(signal) - 1:
            input_2_after = signal[n + 2]
        else:
            input_2_after = 0
        computed_derivative[n] = (-input_2_before - 2 * input_1_before
                                  + 2 * input_1_after + input_2_after) / 8
    return computed_derivative


def square_signal(signal):
    """Compute the square of the signal."""
    squared_signal = np.array(signal) ** 2
    return squared_signal


def apply_moving_window_integration(signal):
    """Apply the moving-window integration to the signal."""
    integrated_signal = np.zeros(len(signal))
    for n in range(len(signal)):
        if n - (WINDOW_SIZE - 1) >= 0:
            window_start = n - (WINDOW_SIZE - 1)
        else:
            window_start = 0
        integrated_signal[n] = np.sum(signal[window_start:n + 1]) / WINDOW_SIZE
    return integrated_signal


def resample_signal(signal, sampling_rate):
    """Resample the signal to the target sampling rate of 200 Hz."""
    resampled_signal = resample_poly(signal, up=TARGET_SAMPLING_RATE,
                                     down=sampling_rate)
    return resampled_signal
