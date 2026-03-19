"""Tests for Half-Time ELO rating feature engineer."""

import numpy as np
import pandas as pd
import pytest


def _make_ht_matches(results):
    """Create matches DataFrame with HT scores from (ht_home, ht_away, ft_home, ft_away) tuples."""
    rows = []
    for i, (ht_h, ht_a, ft_h, ft_a) in enumerate(results):
        rows.append({
            "fixture_id": 2000 + i,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "home_team_id": 1,
            "away_team_id": 2,
            "ht_home": ht_h,
            "ht_away": ht_a,
            "ft_home": ft_h,
            "ft_away": ft_a,
        })
    return pd.DataFrame(rows)


class TestHTEloBasic:
    """Test basic HT Elo functionality."""

    def test_creates_correct_features(self):
        """Should create ht_home_elo, ht_away_elo, ht_elo_diff, ht_win_prob_elo."""
        from src.features.engineers.ratings import HTELORatingFeatureEngineer

        matches = _make_ht_matches([(1, 0, 2, 1), (0, 1, 1, 2)])
        eng = HTELORatingFeatureEngineer()

        result = eng.create_features({"matches": matches})

        expected_cols = {"fixture_id", "ht_home_elo", "ht_away_elo", "ht_elo_diff",
                         "ht_win_prob_elo", "ht_home_elo_sd", "ht_away_elo_sd"}
        assert expected_cols.issubset(set(result.columns))

    def test_ratings_start_at_1500(self):
        """All teams should start at 1500 Elo."""
        from src.features.engineers.ratings import HTELORatingFeatureEngineer

        matches = _make_ht_matches([(1, 0, 2, 1)])
        eng = HTELORatingFeatureEngineer()

        result = eng.create_features({"matches": matches})

        assert result["ht_home_elo"].iloc[0] == 1500.0
        assert result["ht_away_elo"].iloc[0] == 1500.0

    def test_ht_win_updates_rating(self):
        """Team winning at HT should gain Elo."""
        from src.features.engineers.ratings import HTELORatingFeatureEngineer

        # Team 1 wins HT in match 1, then plays again
        matches = _make_ht_matches([(2, 0, 3, 1), (0, 0, 1, 1)])
        eng = HTELORatingFeatureEngineer(k_factor=32.0)

        result = eng.create_features({"matches": matches})

        # After match 1 (HT 2-0 win), home team should have higher Elo
        assert result["ht_home_elo"].iloc[1] > 1500.0
        assert result["ht_away_elo"].iloc[1] < 1500.0

    def test_nan_ht_scores_skipped(self):
        """Matches with NaN HT scores should not update ratings."""
        from src.features.engineers.ratings import HTELORatingFeatureEngineer

        matches = _make_ht_matches([(np.nan, np.nan, 2, 1), (1, 0, 2, 0)])
        eng = HTELORatingFeatureEngineer()

        result = eng.create_features({"matches": matches})

        # After match 1 (NaN HT), ratings should still be 1500
        assert result["ht_home_elo"].iloc[1] == 1500.0
        assert result["ht_away_elo"].iloc[1] == 1500.0

    def test_no_ht_columns_returns_fixture_only(self):
        """If ht_home/ht_away don't exist, return fixture_id only."""
        from src.features.engineers.ratings import HTELORatingFeatureEngineer

        matches = pd.DataFrame({
            "fixture_id": [1, 2],
            "date": pd.date_range("2024-01-01", periods=2),
            "home_team_id": [1, 1],
            "away_team_id": [2, 2],
            "ft_home": [2, 1],
            "ft_away": [1, 2],
        })
        eng = HTELORatingFeatureEngineer()

        result = eng.create_features({"matches": matches})

        assert "fixture_id" in result.columns
        assert len(result.columns) == 1

    def test_draw_gives_equal_update(self):
        """HT draw should give symmetric updates."""
        from src.features.engineers.ratings import HTELORatingFeatureEngineer

        matches = _make_ht_matches([(1, 1, 2, 2), (0, 0, 0, 0)])
        eng = HTELORatingFeatureEngineer(k_factor=32.0, home_advantage=0.0)

        result = eng.create_features({"matches": matches})

        # With home_advantage=0, a draw should keep ratings equal
        np.testing.assert_almost_equal(
            result["ht_home_elo"].iloc[1],
            result["ht_away_elo"].iloc[1],
            decimal=6,
        )

    def test_feature_names(self):
        """get_feature_names should return correct list."""
        from src.features.engineers.ratings import HTELORatingFeatureEngineer

        eng = HTELORatingFeatureEngineer()
        names = eng.get_feature_names()

        assert "ht_home_elo" in names
        assert "ht_elo_diff" in names
        assert "ht_win_prob_elo" in names
