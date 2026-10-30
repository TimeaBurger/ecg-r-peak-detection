"""Preprocess ECG signals for R-peak detection.

Resample the signals to 200 Hz and apply low-pass and high-pass filters
according to the Pan-Tompkins algorithm.
"""

import numpy as np


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
