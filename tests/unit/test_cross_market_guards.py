"""Tests for cross-market NaN guards."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineers.cross_market import CrossMarketFeatureEngineer


class TestSafeGet:
    """Test _safe_get NaN propagation."""

    @pytest.fixture
    def eng(self):
        return CrossMarketFeatureEngineer()

    def test_normal_value(self, eng):
        match = pd.Series({"home_cards_ema": 2.5, "away_cards_ema": 1.5})
        result = eng._safe_get(match, ["home_cards_ema"], 1.5)
        assert result == 2.5

    def test_nan_value_propagates(self, eng):
        match = pd.Series({"home_cards_ema": np.nan})
        result = eng._safe_get(match, ["home_cards_ema"], 1.5)
        assert np.isnan(result)

    def test_column_missing_returns_default(self, eng):
        match = pd.Series({"other_col": 3.0})
        result = eng._safe_get(match, ["home_cards_ema"], 1.5)
        assert result == 1.5

    def test_first_column_nan_does_not_check_second(self, eng):
        """If first column exists but is NaN, return NaN immediately."""
        match = pd.Series({"home_cards_ema": np.nan, "home_avg_cards": 2.0})
        result = eng._safe_get(match, ["home_cards_ema", "home_avg_cards"], 1.5)
        assert np.isnan(result)

    def test_zero_is_valid(self, eng):
        match = pd.Series({"home_cards_ema": 0.0})
        result = eng._safe_get(match, ["home_cards_ema"], 1.5)
        assert result == 0.0


class TestCrossMarketWithNaN:
    """Test that NaN EMAs propagate to interaction features."""

    @pytest.fixture
    def eng(self):
        return CrossMarketFeatureEngineer()

    def test_nan_cards_produces_nan_interactions(self, eng):
        """When card EMAs are NaN, card-related interactions should be NaN."""
        match = pd.Series(
            {
                "fixture_id": 1,
                "home_cards_ema": np.nan,
                "away_cards_ema": np.nan,
                "home_shots_ema": 12.0,
                "away_shots_ema": 10.0,
                "home_corners_ema": 5.0,
                "away_corners_ema": 4.5,
                "home_fouls_committed_ema": 11.0,
                "away_fouls_committed_ema": 12.0,
                "avg_home_open": 2.0,
                "avg_away_open": 3.0,
                "home_avg_yellows": np.nan,
                "away_avg_yellows": np.nan,
            }
        )
        home_cards = eng._safe_get(match, ["home_cards_ema", "home_avg_cards"], 1.5)
        away_cards = eng._safe_get(match, ["away_cards_ema", "away_avg_cards"], 1.5)
        # Both should be NaN
        assert np.isnan(home_cards)
        assert np.isnan(away_cards)
        # Interactions with NaN should produce NaN
        assert np.isnan(home_cards * away_cards)

    def test_valid_data_produces_normal_interactions(self, eng):
        match = pd.Series(
            {
                "fixture_id": 1,
                "home_cards_ema": 2.0,
                "away_cards_ema": 1.5,
            }
        )
        home_cards = eng._safe_get(match, ["home_cards_ema"], 1.5)
        away_cards = eng._safe_get(match, ["away_cards_ema"], 1.5)
        assert home_cards == 2.0
        assert away_cards == 1.5
        assert home_cards * away_cards == pytest.approx(3.0)
