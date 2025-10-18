class DataProcessingError(Exception):
    """Base exception for data processing."""
    pass


class DataValidationError(DataProcessingError):
    """Exception for data validation."""
    pass


class FileLoadError(DataProcessingError):
    """Exception for file loading."""
    pass