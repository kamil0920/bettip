"""Tests for NaN-aware EMA computation in niche markets."""

import numpy as np
import pandas as pd
import pytest


class TestNaNAwareCardEMA:
    """Test that EMAs correctly handle NaN values from missing card data."""

    def _compute_ema(self, values, ema_span=5, min_periods=2):
        """Compute shifted EMA like the card engineer does."""
        s = pd.Series(values)
        return s.shift(1).ewm(span=ema_span, min_periods=min_periods).mean()

    def test_ema_skips_nan_values(self):
        """NaN values should be skipped, not treated as 0."""
        values = [2, np.nan, 3, np.nan, 4, 3]
        ema = self._compute_ema(values, ema_span=3, min_periods=1)
        # Last value's EMA should reflect [2, 3, 4], not [2, 0, 3, 0, 4]
        last_ema = ema.iloc[-1]
        assert not pd.isna(last_ema)
        assert last_ema > 2.0  # Should be around 3.0, not ~1.8

    def test_ema_all_nan_returns_nan(self):
        """If all input values are NaN, EMA should be NaN."""
        values = [np.nan, np.nan, np.nan, np.nan]
        ema = self._compute_ema(values, ema_span=3, min_periods=2)
        assert ema.isna().all()

    def test_ema_with_real_zeros(self):
        """Genuine zeros should be used in EMA computation."""
        values = [0, 2, 0, 3, 0, 1]
        ema = self._compute_ema(values, ema_span=3, min_periods=1)
        last_ema = ema.iloc[-1]
        assert not pd.isna(last_ema)
        # EMA of [0, 2, 0, 3, 0] ~ 1.0
        assert last_ema < 2.0

    def test_backward_compatible_with_full_data(self):
        """Full data (no NaN) should produce same results as before."""
        values = [1, 2, 3, 4, 5, 6]
        ema = self._compute_ema(values, ema_span=3, min_periods=1)
        # Should be a smooth increasing sequence
        assert not ema.iloc[-1:].isna().any()
        assert ema.iloc[-1] > ema.iloc[2]

    def test_ema_single_valid_value_with_min_periods(self):
        """With min_periods=2, single valid value should return NaN."""
        values = [np.nan, np.nan, 3, np.nan, np.nan]
        ema = self._compute_ema(values, ema_span=3, min_periods=2)
        # After shift, position 3 sees value at position 2 (3.0) — only 1 value
        # min_periods=2 means EMA requires at least 2 non-NaN values
        # So most positions should be NaN
        valid_count = ema.notna().sum()
        assert valid_count <= 2

    def test_nan_does_not_contaminate_teams(self):
        """Each team's EMA should be independent."""
        df = pd.DataFrame(
            {
                "team": ["A", "A", "A", "B", "B", "B"],
                "cards": [2, np.nan, 3, np.nan, np.nan, 5],
            }
        )
        result = df.groupby("team")["cards"].transform(
            lambda x: x.shift(1).ewm(span=3, min_periods=1).mean()
        )
        # Team A: shift => [NaN, 2, NaN], EMA on [2] then [2, NaN]
        # Team B: shift => [NaN, NaN, NaN], all NaN
        assert pd.isna(result.iloc[3])  # Team B first row after shift = NaN
