"""Unit tests for the visualization module."""

import unittest

import pandas as pd

from rpeak_detection.visualization import create_hrv_scatter_plot

EXAMPLE_RESULTS = pd.DataFrame([
    {
        "detector_name": "pan_tompkins",
        "sdnn": 50.0,
        "reference_sdnn": 48.0,
    },
    {
        "detector_name": "baseline",
        "sdnn": 650.0,
        "reference_sdnn": 48.0,
    },
])


class TestCreateHrvScatterPlot(unittest.TestCase):
    """Tests for create_hrv_scatter_plot."""

    def test_use_metric_in_title(self):
        """Show the upper-case metric name in the title."""
        plot = create_hrv_scatter_plot(EXAMPLE_RESULTS, "sdnn")
        self.assertEqual(plot.title.text, "SDNN: detected versus reference")

    def test_draw_both_detectors_and_identity_line(self):
        """Draw two groups of points and one line."""
        plot = create_hrv_scatter_plot(EXAMPLE_RESULTS, "sdnn")
        self.assertEqual(len(plot.renderers), 3)
