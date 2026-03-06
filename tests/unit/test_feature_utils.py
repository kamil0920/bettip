"""Tests for shared feature engineering utilities."""

import numpy as np
import pandas as pd
import pytest

from src.features.utils import MIN_FOR_HURST, is_degenerate, safe_rolling_apply


class TestSafeRollingApply:
    def test_sufficient_data_computes(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = safe_rolling_apply(s, window=5, func=np.mean, min_valid=3)
        # Last value: window [6,7,8,9,10], mean=8.0
        assert result.iloc[-1] == pytest.approx(8.0)

    def test_insufficient_data_returns_nan(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = safe_rolling_apply(s, window=5, func=np.mean, min_valid=5)
        # Only 3 values in window, need 5 -> NaN
        assert np.isnan(result.iloc[-1])

    def test_nan_in_window_reduces_valid_count(self):
        s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        result = safe_rolling_apply(s, window=5, func=np.mean, min_valid=3)
        # 3 valid values [1, 3, 5] -> passes min_valid=3
        assert result.iloc[-1] == pytest.approx(3.0)

    def test_all_nan_window(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        result = safe_rolling_apply(s, window=3, func=np.mean, min_valid=1)
        assert np.isnan(result.iloc[-1])

    def test_rolling_window_boundary(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_rolling_apply(s, window=5, func=np.mean, min_valid=4)
        # Position 2 (3rd element): window has [1, 2, 3] -> only 3 valid, need 4
        assert np.isnan(result.iloc[2])
        # Position 4: window has [1, 2, 3, 4, 5] -> 5 valid >= 4
        assert result.iloc[4] == pytest.approx(3.0)


class TestIsDegenerate:
    def test_constant_values(self):
        assert is_degenerate(np.array([3.0, 3.0, 3.0, 3.0]))

    def test_varied_values(self):
        assert not is_degenerate(np.array([3.0, 3.0, 3.0, 4.0]))

    def test_all_nan(self):
        assert is_degenerate(np.array([np.nan, np.nan]))

    def test_single_value_with_nan(self):
        assert is_degenerate(np.array([5.0, np.nan, np.nan]))

    def test_empty(self):
        assert is_degenerate(np.array([]))
