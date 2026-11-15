"""Compute time-domain heart rate variability (HRV) metrics.

The metrics SDNN and RMSSD are computed from the intervals between
beat positions.
"""

import numpy as np


def compute_rr_intervals(beat_positions, sampling_rate):
    """Return the intervals between the beat positions in milliseconds.

    The beat positions are given in samples and the sampling rate in Hz.
    """
    intervals_in_samples = np.diff(beat_positions)
    return (intervals_in_samples / sampling_rate) * 1000


def compute_sdnn(rr_intervals):
    """Return the standard deviation of the RR intervals in milliseconds.

    The standard deviation is divided by N, since the metric describes the
    recording rather than estimating a population value. The same formula
    is applied to annotated and detected beats.
    """
    return np.std(rr_intervals)


def compute_rmssd(rr_intervals):
    """Return the RMSSD of the RR intervals in milliseconds.

    The RMSSD is the square root of the mean of the squared differences
    between successive RR intervals (Task Force, 1996).
    """
    successive_differences = np.diff(rr_intervals)
    return np.sqrt(np.mean(successive_differences ** 2))
