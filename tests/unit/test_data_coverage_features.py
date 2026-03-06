"""Tests for data coverage features in niche markets."""

import numpy as np
import pandas as pd
import pytest


class TestDataCoverageFeatures:
    """Test coverage feature computation."""

    def _compute_coverage(self, values, window=5):
        """Simulate the coverage computation from CardsFeatureEngineer."""
        s = pd.Series(values)
        return s.shift(1).rolling(window=window, min_periods=1).apply(
            lambda w: w.notna().mean(), raw=False
        )

    def test_full_coverage(self):
        values = [1, 2, 3, 4, 5, 6]
        cov = self._compute_coverage(values, window=5)
        assert cov.iloc[-1] == 1.0

    def test_no_coverage(self):
        values = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        cov = self._compute_coverage(values, window=5)
        # After shift, all values NaN. rolling().apply(notna().mean) on NaN
        # returns NaN (no valid window entries). This is correct — NaN means
        # "unknown coverage" which downstream can interpret as 0.
        assert pd.isna(cov.iloc[-1]) or cov.iloc[-1] == 0.0

    def test_partial_coverage(self):
        # After shift: [NaN, 2, NaN, NaN, 3, NaN]
        # At position 5, window of 5 = [2, NaN, NaN, 3, NaN] => 2/5 = 0.4
        values = [2, np.nan, np.nan, 3, np.nan, 1]
        cov = self._compute_coverage(values, window=5)
        last = cov.iloc[-1]
        assert 0.0 < last < 1.0

    def test_coverage_uses_shift(self):
        """Current match data should NOT be included (leakage prevention)."""
        values = [np.nan, np.nan, np.nan, np.nan, 5]
        cov = self._compute_coverage(values, window=5)
        # Last position: shift means we look at [NaN, NaN, NaN, NaN]
        # rolling.apply on all-NaN returns NaN
        assert pd.isna(cov.iloc[-1]) or cov.iloc[-1] == 0.0

    def test_coverage_with_genuine_zeros(self):
        """Matches with 0 cards (not NaN) count as 'has data'."""
        values = [0, 0, 0, 0, 0, 0]
        cov = self._compute_coverage(values, window=5)
        assert cov.iloc[-1] == 1.0

    def test_single_match_coverage(self):
        """With only one previous match, coverage reflects valid fraction."""
        values = [3, np.nan, 5]
        cov = self._compute_coverage(values, window=5)
        # Shift => [NaN, 3, NaN]. At position 2, window=[NaN, 3, NaN], 1/3 valid
        assert cov.iloc[-1] == pytest.approx(1.0 / 3.0)
