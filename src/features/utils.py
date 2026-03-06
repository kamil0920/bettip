"""Shared utilities for defensive feature engineering."""

import numpy as np
import pandas as pd

# Minimum non-NaN observations required for each computation type
MIN_FOR_EMA = 3
MIN_FOR_ROLLING_STD = 4
MIN_FOR_SKEWNESS = 5
MIN_FOR_KURTOSIS = 5
MIN_FOR_HURST = 10
MIN_FOR_ENTROPY = 8


def safe_rolling_apply(
    series: pd.Series, window: int, func, min_valid: int
) -> pd.Series:
    """Apply a rolling function only when enough non-NaN values exist.

    Returns NaN for windows with fewer than min_valid non-NaN observations.
    """

    def guarded_func(x):
        valid = x[~np.isnan(x)]
        if len(valid) < min_valid:
            return np.nan
        return func(valid)

    return series.rolling(window=window, min_periods=1).apply(guarded_func, raw=True)


def is_degenerate(values: np.ndarray, min_unique: int = 2) -> bool:
    """Check if values are too uniform for meaningful statistics."""
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return True
    return len(np.unique(clean)) < min_unique
