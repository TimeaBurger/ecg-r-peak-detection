"""Compute time-domain heart rate variability (HRV) metrics.

The metrics SDNN and RMSSD are computed from the intervals between
beat positions.
"""

import numpy as np


def compute_rr_intervals(beat_positions, sampling_rate):
    """Return the intervals between the beat positions in milliseconds."""
    intervals_in_samples = np.diff(beat_positions)
    return (intervals_in_samples / sampling_rate) * 1000


def compute_sdnn(rr_intervals):
    """Return the standard deviation of the RR intervals in milliseconds."""
    return np.std(rr_intervals)


def compute_rmssd(rr_intervals):
    """Return the RMSSD of the RR intervals.

    Root mean square of successive differences between RR
    intervals in milliseconds.
    """
    successive_differences = np.diff(rr_intervals)
    return np.sqrt(np.mean(successive_differences ** 2))
