"""Tests for Kelly Index computation and match difficulty classification."""
import numpy as np
import pandas as pd
import pytest

from src.odds.odds_features import compute_kelly_index, remove_margin_shin_3way


class TestRemoveMarginShin3Way:
    """Tests for Shin's margin removal method."""

    def test_fair_odds_no_margin(self):
        """When there's no overround, Shin should return normalized probs."""
        # These are roughly fair odds (sum of implied probs ≈ 1.0)
        p_h, p_d, p_a = remove_margin_shin_3way(2.5, 3.3, 3.0)
        assert abs(p_h + p_d + p_a - 1.0) < 1e-6

    def test_probabilities_sum_to_one(self):
        """Shin probabilities must always sum to 1.0."""
        # Typical bookmaker odds with ~5% margin
        p_h, p_d, p_a = remove_margin_shin_3way(1.80, 3.60, 4.50)
        assert abs(p_h + p_d + p_a - 1.0) < 1e-6

    def test_favourite_gets_higher_probability(self):
        """Favourite (lowest odds) should have highest probability."""
        p_h, p_d, p_a = remove_margin_shin_3way(1.50, 4.00, 7.00)
        assert p_h > p_d
        assert p_h > p_a

    def test_shin_vs_multiplicative_differs(self):
        """Shin should differ from multiplicative normalization."""
        odds_h, odds_d, odds_a = 1.40, 4.50, 9.00
        q = np.array([1 / odds_h, 1 / odds_d, 1 / odds_a])
        q_mult = q / q.sum()

        p_h, p_d, p_a = remove_margin_shin_3way(odds_h, odds_d, odds_a)

        # Shin corrects for favourite-longshot bias by attributing part of
        # the overround to insider trading. This gives the favourite a higher
        # probability than multiplicative (longshots are overpriced).
        assert p_h > q_mult[0], "Shin should give favourite higher prob than multiplicative"
        # Longshot should get lower probability
        assert p_a < q_mult[2], "Shin should give longshot lower prob than multiplicative"

    def test_all_probabilities_positive(self):
        """All probabilities should be strictly positive."""
        p_h, p_d, p_a = remove_margin_shin_3way(1.20, 6.00, 15.00)
        assert p_h > 0
        assert p_d > 0
        assert p_a > 0

    def test_invalid_odds_returns_uniform(self):
        """Invalid odds (<= 1.0) should return uniform distribution."""
        p_h, p_d, p_a = remove_margin_shin_3way(0.5, 3.0, 4.0)
        assert abs(p_h - 1 / 3) < 1e-6
        assert abs(p_d - 1 / 3) < 1e-6
        assert abs(p_a - 1 / 3) < 1e-6

    def test_known_bookmaker_margin(self):
        """Test with realistic bookmaker odds (Bet365 style)."""
        # B365: Home 2.10, Draw 3.40, Away 3.50 (typical balanced match)
        p_h, p_d, p_a = remove_margin_shin_3way(2.10, 3.40, 3.50)
        assert abs(p_h + p_d + p_a - 1.0) < 1e-6
        # Home should still be favourite
        assert p_h > p_d
        assert p_h > p_a


class TestComputeKellyIndex:
    """Tests for Kelly Index match difficulty classification."""

    def _make_df(self, n=5):
        """Create a test DataFrame with bookmaker odds."""
        np.random.seed(42)
        return pd.DataFrame({
            'b365_home_close': np.random.uniform(1.3, 5.0, n),
            'b365_draw_close': np.random.uniform(3.0, 4.5, n),
            'b365_away_close': np.random.uniform(1.5, 8.0, n),
            'ps_home_close': np.random.uniform(1.3, 5.0, n),
            'ps_draw_close': np.random.uniform(3.0, 4.5, n),
            'ps_away_close': np.random.uniform(1.5, 8.0, n),
        })

    def test_output_columns_present(self):
        """All expected output columns should be present."""
        df = self._make_df()
        result = compute_kelly_index(df)
        for col in ['kelly_index_home_avg', 'kelly_index_away_avg',
                     'kelly_index_draw_avg', 'kelly_f99',
                     'match_difficulty_type', 'n_bookmakers_k_above_1']:
            assert col in result.columns, f"Missing column: {col}"

    def test_difficulty_types_valid(self):
        """Match difficulty should be 1, 2, or 3."""
        df = self._make_df(100)
        result = compute_kelly_index(df)
        assert set(result['match_difficulty_type'].unique()).issubset({1, 2, 3})

    def test_f99_realistic_odds(self):
        """Market return rate should be below 1.0 with realistic bookmaker odds."""
        # Realistic odds: overround ~5%
        df = pd.DataFrame({
            'b365_home_close': [2.10, 1.50],
            'b365_draw_close': [3.40, 4.20],
            'b365_away_close': [3.50, 6.50],
            'ps_home_close': [2.15, 1.52],
            'ps_draw_close': [3.30, 4.10],
            'ps_away_close': [3.40, 6.20],
        })
        result = compute_kelly_index(df)
        assert (result['kelly_f99'].dropna() < 1.0).all()

    def test_no_bookmaker_columns_returns_defaults(self):
        """When no bookmaker columns exist, should return NaN/defaults."""
        df = pd.DataFrame({'some_col': [1, 2, 3]})
        result = compute_kelly_index(df)
        assert result['match_difficulty_type'].eq(3).all()
        assert result['kelly_index_home_avg'].isna().all()

    def test_avg_odds_fallback(self):
        """Should fall back to avg_* columns when no individual bookmaker columns."""
        df = pd.DataFrame({
            'avg_home_close': [1.50, 2.50, 4.00],
            'avg_draw_close': [4.00, 3.30, 3.50],
            'avg_away_close': [6.00, 2.80, 1.80],
        })
        result = compute_kelly_index(df)
        assert result['kelly_f99'].notna().all()

    def test_type1_bookmaker_disagreement(self):
        """When bookmakers disagree significantly, K > 1 for at least one."""
        # Bookmakers with significant disagreement trigger K > 1
        df = pd.DataFrame({
            'b365_home_close': [1.50, 2.00],
            'b365_draw_close': [4.00, 3.50],
            'b365_away_close': [7.00, 3.80],
            'ps_home_close': [1.80, 2.00],
            'ps_draw_close': [3.20, 3.50],
            'ps_away_close': [4.50, 3.80],
            'wh_home_close': [1.55, 2.00],
            'wh_draw_close': [3.80, 3.50],
            'wh_away_close': [6.50, 3.80],
        })
        result = compute_kelly_index(df)
        # First match has big disagreement on away odds → should get at least 1 K > 1
        assert result.iloc[0]['n_bookmakers_k_above_1'] >= 1
        # Second match: all bookmakers agree → Type 3
        assert result.iloc[1]['match_difficulty_type'] == 3
