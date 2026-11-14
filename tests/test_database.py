"""Unit tests for the database module."""

import os
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from rpeak_detection.database import load_results, save_results


class TestSaveAndLoadResults(unittest.TestCase):
    """Tests for save_results and load_results."""

    def test_load_returns_saved_results(self):
        """Return the same table that was saved."""
        results = pd.DataFrame([
            {"record_name": "100", "true_positives": 5},
            {"record_name": "101", "true_positives": 7},
        ])
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "test.db")
            save_results(results, database_path)
            loaded = load_results(database_path)
        assert_frame_equal(loaded, results)
