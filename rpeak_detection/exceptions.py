"""Custom exceptions for the rpeak_detection package."""


class RPeakDetectionError(Exception):
    """Base class for all exceptions in the rpeak_detection package."""


class RecordNotFoundError(RPeakDetectionError):
    """Raised when a record cannot be found in the data directory."""


class ChannelNotAvailableError(RPeakDetectionError):
    """Raised when a channel is not available in the recording."""
