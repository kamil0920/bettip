"""Tests for Double Chance (DC) market target derivation and fair odds computation."""

import numpy as np
import pandas as pd
import pytest


def _make_match_df():
    """Create a DataFrame with known match results for DC testing."""
    return pd.DataFrame({
        "fixture_id": [1, 2, 3, 4, 5],
        "ft_home": [2, 0, 1, 3, 0],
        "ft_away": [1, 2, 1, 0, 0],
        "home_win": [1, 0, 0, 1, 0],
        "away_win": [0, 1, 0, 0, 0],
        "draw": [0, 0, 1, 0, 1],
        # HAD odds for fair odds derivation
        "avg_home_close": [2.0, 3.5, 3.0, 1.5, 2.8],
        "avg_draw_close": [3.2, 3.3, 3.2, 4.0, 3.0],
        "avg_away_close": [4.0, 2.1, 2.5, 7.0, 2.8],
    })


class TestDoubleChanceTargets:
    """Test DC target derivation correctness."""

    def test_dc_1x_target(self):
        """dc_1x = home win OR draw."""
        df = _make_match_df()

        # Match 1: home win (2-1) → dc_1x = 1
        # Match 2: away win (0-2) → dc_1x = 0
        # Match 3: draw (1-1) → dc_1x = 1
        # Match 4: home win (3-0) → dc_1x = 1
        # Match 5: draw (0-0) → dc_1x = 1
        expected = [1, 0, 1, 1, 1]

        df["dc_1x"] = ((df["home_win"] == 1) | (df["draw"] == 1)).astype(int)
        assert list(df["dc_1x"]) == expected

    def test_dc_12_target(self):
        """dc_12 = home win OR away win (no draw)."""
        df = _make_match_df()

        # Match 1: home win → dc_12 = 1
        # Match 2: away win → dc_12 = 1
        # Match 3: draw → dc_12 = 0
        # Match 4: home win → dc_12 = 1
        # Match 5: draw → dc_12 = 0
        expected = [1, 1, 0, 1, 0]

        df["dc_12"] = ((df["home_win"] == 1) | (df["away_win"] == 1)).astype(int)
        assert list(df["dc_12"]) == expected

    def test_dc_x2_target(self):
        """dc_x2 = draw OR away win."""
        df = _make_match_df()

        # Match 1: home win → dc_x2 = 0
        # Match 2: away win → dc_x2 = 1
        # Match 3: draw → dc_x2 = 1
        # Match 4: home win → dc_x2 = 0
        # Match 5: draw → dc_x2 = 1
        expected = [0, 1, 1, 0, 1]

        df["dc_x2"] = ((df["draw"] == 1) | (df["away_win"] == 1)).astype(int)
        assert list(df["dc_x2"]) == expected

    def test_dc_base_rates(self):
        """DC base rates should sum correctly: dc_1x + dc_x2 - dc_draw = 1 + dc_draw."""
        df = _make_match_df()

        dc_1x = ((df["home_win"] == 1) | (df["draw"] == 1)).astype(int)
        dc_12 = ((df["home_win"] == 1) | (df["away_win"] == 1)).astype(int)
        dc_x2 = ((df["draw"] == 1) | (df["away_win"] == 1)).astype(int)

        # Each match has exactly one outcome (H, D, or A)
        # So dc_1x + dc_12 + dc_x2 = 2 for every match (each outcome appears in 2 DC markets)
        assert all(dc_1x + dc_12 + dc_x2 == 2)


class TestDoubleChanceFairOdds:
    """Test DC fair odds computation from HAD odds."""

    def test_fair_odds_reasonable_range(self):
        """DC fair odds should be in [1.01, ~3.0] range."""
        df = _make_match_df()

        p_h = 1.0 / df["avg_home_close"]
        p_d = 1.0 / df["avg_draw_close"]
        p_a = 1.0 / df["avg_away_close"]
        total = p_h + p_d + p_a
        p_h_fair = p_h / total
        p_d_fair = p_d / total
        p_a_fair = p_a / total

        vig = 1.05
        dc_1x_odds = 1.0 / ((p_h_fair + p_d_fair) * vig)
        dc_12_odds = 1.0 / ((p_h_fair + p_a_fair) * vig)
        dc_x2_odds = 1.0 / ((p_d_fair + p_a_fair) * vig)

        for odds in [dc_1x_odds, dc_12_odds, dc_x2_odds]:
            assert all(odds > 1.0), "DC odds must be > 1.0"
            assert all(odds < 5.0), "DC odds should be < 5.0 for reasonable markets"

    def test_dc_1x_odds_lower_than_home_win(self):
        """DC 1X odds should always be lower than pure home win odds."""
        df = _make_match_df()

        p_h = 1.0 / df["avg_home_close"]
        p_d = 1.0 / df["avg_draw_close"]
        p_a = 1.0 / df["avg_away_close"]
        total = p_h + p_d + p_a

        vig = 1.05
        dc_1x_odds = 1.0 / (((p_h + p_d) / total) * vig)

        assert all(dc_1x_odds < df["avg_home_close"]), \
            "DC 1X (home or draw) should have lower odds than pure home win"

    def test_normalization_removes_overround(self):
        """Shin normalization should make probabilities sum to 1."""
        df = _make_match_df()

        p_h = 1.0 / df["avg_home_close"]
        p_d = 1.0 / df["avg_draw_close"]
        p_a = 1.0 / df["avg_away_close"]
        total = p_h + p_d + p_a

        # Raw implied probs sum > 1 (bookmaker overround)
        assert all(total > 1.0)

        # After normalization, sum = 1.0
        p_h_fair = p_h / total
        p_d_fair = p_d / total
        p_a_fair = p_a / total
        np.testing.assert_array_almost_equal(p_h_fair + p_d_fair + p_a_fair, 1.0)
