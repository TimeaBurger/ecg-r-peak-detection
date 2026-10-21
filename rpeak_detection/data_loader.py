"""Load ECG recordings from a local copy of the MIT-BIH Arrhythmia
Database and provide their signals and annotations.
"""

import wfdb


class ECGRecord:
    """Hold the signal and expert annotations of a single ECG recording."""

    def __init__(self, record_name, signal, sampling_rate, annotated_positions,
                 annotated_symbols):
        """Create a record from the given name, signal and annotations."""
        self.record_name = record_name
        self.signal = signal
        self.sampling_rate = sampling_rate
        self.annotated_positions = annotated_positions
        self.annotated_symbols = annotated_symbols


class ECGRecordLoader:
    """Load ECG recordings."""

    def __init__(self, data_dir="data/mitdb", channel="MLII"):
        """Create a loader for the given data directory and channel."""
        self.data_dir = data_dir
        self.channel = channel

    def load_record(self, record_name):
        """Load the recording with the given record name."""
        record_path = f"{self.data_dir}/{record_name}"
        record = wfdb.rdrecord(record_path)
        annotations = wfdb.rdann(record_path, "atr")
        channel_index = record.sig_name.index(self.channel)
        signal = record.p_signal[:, channel_index]
        return ECGRecord(record_name, signal, record.fs, annotations.sample,
                         annotations.symbol)
