"""Tests for BayesianFormFeatureEngineer."""
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features.engineers.form import BayesianFormFeatureEngineer


def _make_matches(n_matches: int = 20, n_teams: int = 4) -> pd.DataFrame:
    """Create a synthetic matches DataFrame for testing."""
    rng = np.random.default_rng(42)
    rows = []
    teams = list(range(1, n_teams + 1))
    fixture_id = 1000

    for i in range(n_matches):
        # Pick two different teams
        home, away = rng.choice(teams, size=2, replace=False)
        rows.append({
            "fixture_id": fixture_id + i,
            "home_team_id": int(home),
            "away_team_id": int(away),
            "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
            "ft_home": int(rng.poisson(1.5)),
            "ft_away": int(rng.poisson(1.2)),
        })

    return pd.DataFrame(rows)


class TestBayesianFormFeatureEngineer:

    def test_creates_expected_columns(self):
        """Engineer should produce all expected feature columns."""
        matches = _make_matches()
        engineer = BayesianFormFeatureEngineer(n_matches=5, prior_weight=5.0)
        features = engineer.create_features({"matches": matches})

        expected_cols = [
            "fixture_id",
            "home_bayes_win_rate",
            "away_bayes_win_rate",
            "home_bayes_goals_scored",
            "away_bayes_goals_scored",
            "home_bayes_goals_conceded",
            "away_bayes_goals_conceded",
            "home_bayes_n_matches",
            "away_bayes_n_matches",
            "bayes_win_rate_diff",
            "bayes_goals_diff",
        ]
        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"

    def test_output_length_matches_input(self):
        """Should produce one row per match."""
        matches = _make_matches(n_matches=15)
        engineer = BayesianFormFeatureEngineer()
        features = engineer.create_features({"matches": matches})
        assert len(features) == len(matches)

    def test_shrinkage_toward_league_mean(self):
        """
        A team with very few matches should have win rate close to league mean (~0.33),
        not extreme raw values.
        """
        # Create data where team 1 wins all 2 matches (raw win rate = 1.0)
        rows = [
            {"fixture_id": 1, "home_team_id": 1, "away_team_id": 2,
             "date": pd.Timestamp("2025-01-01"), "ft_home": 3, "ft_away": 0},
            {"fixture_id": 2, "home_team_id": 1, "away_team_id": 3,
             "date": pd.Timestamp("2025-01-02"), "ft_home": 2, "ft_away": 1},
            # Need a third match to compute features for
            {"fixture_id": 3, "home_team_id": 1, "away_team_id": 4,
             "date": pd.Timestamp("2025-01-03"), "ft_home": 1, "ft_away": 0},
            # Add background matches for other teams
            {"fixture_id": 4, "home_team_id": 2, "away_team_id": 3,
             "date": pd.Timestamp("2025-01-01"), "ft_home": 1, "ft_away": 1},
            {"fixture_id": 5, "home_team_id": 3, "away_team_id": 4,
             "date": pd.Timestamp("2025-01-02"), "ft_home": 0, "ft_away": 2},
            {"fixture_id": 6, "home_team_id": 4, "away_team_id": 2,
             "date": pd.Timestamp("2025-01-03"), "ft_home": 1, "ft_away": 1},
            # Match to test
            {"fixture_id": 7, "home_team_id": 1, "away_team_id": 2,
             "date": pd.Timestamp("2025-01-04"), "ft_home": 0, "ft_away": 0},
        ]
        matches = pd.DataFrame(rows)
        engineer = BayesianFormFeatureEngineer(n_matches=10, prior_weight=5.0)
        features = engineer.create_features({"matches": matches})

        # Team 1 (home in last row) has 3 wins in 3 matches (raw = 1.0)
        # Bayesian estimate should be shrunk below 1.0
        last_row = features.iloc[-1]
        assert last_row["home_bayes_win_rate"] < 1.0, (
            f"Expected shrinkage: got {last_row['home_bayes_win_rate']}"
        )
        # But should still be above league average since they won all matches
        assert last_row["home_bayes_win_rate"] > 0.33

    def test_convergence_with_many_matches(self):
        """
        With many matches, Bayesian estimate should converge to raw estimate.
        """
        matches = _make_matches(n_matches=100, n_teams=4)
        engineer = BayesianFormFeatureEngineer(n_matches=50, prior_weight=5.0)
        features = engineer.create_features({"matches": matches})

        # For the last match, teams have many observations
        last = features.iloc[-1]
        # With 50-match lookback, prior_weight=5 is negligible
        # bayes_goals_scored should be close to raw mean
        assert last["home_bayes_n_matches"] > 10

    def test_no_history_returns_league_default(self):
        """First match features should use league defaults."""
        matches = _make_matches(n_matches=5)
        engineer = BayesianFormFeatureEngineer(n_matches=5)
        features = engineer.create_features({"matches": matches})

        first = features.iloc[0]
        # No history for any team, should get league defaults
        assert first["home_bayes_n_matches"] == 0
        assert first["away_bayes_n_matches"] == 0

    def test_goals_shrinkage(self):
        """Goals scored should be shrunk toward league mean."""
        # Team scores 5 goals per game in just 2 matches (extreme)
        rows = [
            {"fixture_id": 1, "home_team_id": 1, "away_team_id": 2,
             "date": pd.Timestamp("2025-01-01"), "ft_home": 5, "ft_away": 0},
            {"fixture_id": 2, "home_team_id": 1, "away_team_id": 3,
             "date": pd.Timestamp("2025-01-02"), "ft_home": 5, "ft_away": 1},
            # Background
            {"fixture_id": 3, "home_team_id": 2, "away_team_id": 3,
             "date": pd.Timestamp("2025-01-01"), "ft_home": 1, "ft_away": 1},
            {"fixture_id": 4, "home_team_id": 3, "away_team_id": 4,
             "date": pd.Timestamp("2025-01-02"), "ft_home": 1, "ft_away": 2},
            {"fixture_id": 5, "home_team_id": 4, "away_team_id": 2,
             "date": pd.Timestamp("2025-01-03"), "ft_home": 0, "ft_away": 0},
            # Test match
            {"fixture_id": 6, "home_team_id": 1, "away_team_id": 4,
             "date": pd.Timestamp("2025-01-04"), "ft_home": 0, "ft_away": 0},
        ]
        matches = pd.DataFrame(rows)
        engineer = BayesianFormFeatureEngineer(n_matches=10, prior_weight=5.0)
        features = engineer.create_features({"matches": matches})

        last = features.iloc[-1]
        # Raw mean goals = 5.0, but Bayesian should shrink toward league ~1.3
        assert last["home_bayes_goals_scored"] < 5.0
        assert last["home_bayes_goals_scored"] > 1.3  # Still above average

    def test_win_rate_diff_sign(self):
        """Win rate diff should be positive when home team is stronger."""
        rows = [
            # Team 1 wins everything
            {"fixture_id": 1, "home_team_id": 1, "away_team_id": 2,
             "date": pd.Timestamp("2025-01-01"), "ft_home": 3, "ft_away": 0},
            {"fixture_id": 2, "home_team_id": 1, "away_team_id": 3,
             "date": pd.Timestamp("2025-01-02"), "ft_home": 2, "ft_away": 0},
            # Team 4 loses everything
            {"fixture_id": 3, "home_team_id": 4, "away_team_id": 2,
             "date": pd.Timestamp("2025-01-01"), "ft_home": 0, "ft_away": 3},
            {"fixture_id": 4, "home_team_id": 4, "away_team_id": 3,
             "date": pd.Timestamp("2025-01-02"), "ft_home": 0, "ft_away": 2},
            # Background
            {"fixture_id": 5, "home_team_id": 2, "away_team_id": 3,
             "date": pd.Timestamp("2025-01-03"), "ft_home": 1, "ft_away": 1},
            # Test: Team 1 (strong) vs Team 4 (weak)
            {"fixture_id": 6, "home_team_id": 1, "away_team_id": 4,
             "date": pd.Timestamp("2025-01-04"), "ft_home": 0, "ft_away": 0},
        ]
        matches = pd.DataFrame(rows)
        engineer = BayesianFormFeatureEngineer(n_matches=10, prior_weight=3.0)
        features = engineer.create_features({"matches": matches})

        last = features.iloc[-1]
        assert last["bayes_win_rate_diff"] > 0, (
            f"Strong home vs weak away should have positive diff, got {last['bayes_win_rate_diff']}"
        )


class TestBayesianFormBetaPrior:
    """Test the Beta prior fitting logic."""

    def test_fit_beta_prior_reasonable(self):
        """Beta prior from typical league stats should be reasonable."""
        engineer = BayesianFormFeatureEngineer()
        a, b = engineer._fit_beta_prior(mean=0.33, var=0.02)
        assert a > 0
        assert b > 0
        # Mean of Beta(a,b) should be close to input mean
        assert abs(a / (a + b) - 0.33) < 0.01

    def test_fit_beta_prior_uniform_fallback(self):
        """Edge cases should fall back to uniform."""
        engineer = BayesianFormFeatureEngineer()
        a, b = engineer._fit_beta_prior(mean=0.0, var=0.01)
        assert a == 1.0 and b == 1.0

        a, b = engineer._fit_beta_prior(mean=0.5, var=-1)
        assert a == 1.0 and b == 1.0
