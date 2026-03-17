"""Tests for Pi-ratings feature engineer."""
import numpy as np
import pandas as pd
import pytest

from src.features.engineers.ratings import PiRatingFeatureEngineer


def _make_matches(n=20):
    """Create synthetic match data for testing."""
    np.random.seed(42)
    teams = [101, 102, 103, 104]
    matches = []
    base_date = pd.Timestamp("2024-01-01")
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        matches.append({
            'fixture_id': 1000 + i,
            'date': base_date + pd.Timedelta(days=i * 3),
            'home_team_id': home,
            'away_team_id': away,
            'ft_home': np.random.randint(0, 4),
            'ft_away': np.random.randint(0, 3),
        })
    return pd.DataFrame(matches)


class TestPiRatingFeatureEngineer:
    """Tests for Pi-rating computation."""

    def test_output_columns(self):
        """Pi-rating engineer should produce expected columns."""
        engineer = PiRatingFeatureEngineer()
        df = engineer.create_features({'matches': _make_matches()})
        expected = [
            'fixture_id', 'home_pi_rating_h', 'home_pi_rating_a',
            'away_pi_rating_h', 'away_pi_rating_a',
            'pi_rating_diff', 'pi_rating_home_advantage',
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_initial_ratings_zero(self):
        """First match should have all ratings at 0.0."""
        engineer = PiRatingFeatureEngineer()
        df = engineer.create_features({'matches': _make_matches()})
        first = df.iloc[0]
        assert first['home_pi_rating_h'] == 0.0
        assert first['home_pi_rating_a'] == 0.0
        assert first['away_pi_rating_h'] == 0.0
        assert first['away_pi_rating_a'] == 0.0

    def test_ratings_update_after_match(self):
        """Ratings should change after the first match."""
        engineer = PiRatingFeatureEngineer()
        matches = _make_matches(5)
        # Ensure first match has a decisive result
        matches.loc[0, 'ft_home'] = 3
        matches.loc[0, 'ft_away'] = 0
        df = engineer.create_features({'matches': matches})

        # Second match involving the same home team should have non-zero ratings
        home_team_first = matches.loc[0, 'home_team_id']
        # Find next match where this team plays
        for i in range(1, len(df)):
            match = matches.iloc[i]
            if match['home_team_id'] == home_team_first:
                assert df.iloc[i]['home_pi_rating_h'] != 0.0
                break
            elif match['away_team_id'] == home_team_first:
                assert df.iloc[i]['away_pi_rating_h'] != 0.0 or df.iloc[i]['away_pi_rating_a'] != 0.0
                break

    def test_goal_diff_dampening(self):
        """Large blowouts should not dominate — dampening via log10."""
        engineer = PiRatingFeatureEngineer(lambda_=0.1, c=3.0)

        # Match 1: 5-0 blowout
        matches_blowout = pd.DataFrame([
            {'fixture_id': 1, 'date': '2024-01-01', 'home_team_id': 1, 'away_team_id': 2, 'ft_home': 5, 'ft_away': 0},
            {'fixture_id': 2, 'date': '2024-01-04', 'home_team_id': 1, 'away_team_id': 3, 'ft_home': 0, 'ft_away': 0},
        ])

        # Match 1: 1-0 narrow win
        matches_narrow = pd.DataFrame([
            {'fixture_id': 1, 'date': '2024-01-01', 'home_team_id': 1, 'away_team_id': 2, 'ft_home': 1, 'ft_away': 0},
            {'fixture_id': 2, 'date': '2024-01-04', 'home_team_id': 1, 'away_team_id': 3, 'ft_home': 0, 'ft_away': 0},
        ])

        df_blowout = engineer.create_features({'matches': matches_blowout})
        df_narrow = engineer.create_features({'matches': matches_narrow})

        # Rating after 5-0 should be higher than after 1-0, but NOT 5x higher
        rating_blowout = df_blowout.iloc[1]['home_pi_rating_h']
        rating_narrow = df_narrow.iloc[1]['home_pi_rating_h']

        assert rating_blowout > rating_narrow, "5-0 should give higher rating than 1-0"
        ratio = rating_blowout / rating_narrow if rating_narrow != 0 else float('inf')
        assert ratio < 3.0, f"Dampening should prevent 5-0 from being >3x of 1-0 (got {ratio:.2f}x)"

    def test_cross_learning(self):
        """Away rating should move toward home rating (gamma effect)."""
        engineer = PiRatingFeatureEngineer(lambda_=0.1, gamma=0.70)
        matches = pd.DataFrame([
            {'fixture_id': 1, 'date': '2024-01-01', 'home_team_id': 1, 'away_team_id': 2, 'ft_home': 2, 'ft_away': 0},
            {'fixture_id': 2, 'date': '2024-01-04', 'home_team_id': 1, 'away_team_id': 3, 'ft_home': 0, 'ft_away': 0},
        ])
        df = engineer.create_features({'matches': matches})

        # After team 1 wins at home, home rating goes up
        # Cross-learning: away rating should also move (toward home rating direction)
        home_h = df.iloc[1]['home_pi_rating_h']
        home_a = df.iloc[1]['home_pi_rating_a']

        # Both should be positive (team won), away should be less than home
        assert home_h > 0, "Home rating should increase after home win"
        # Away rating should be in same direction as home (gamma > 0)
        if home_h > 0:
            assert home_a > 0, "Cross-learning should push away rating in same direction as home"
            assert home_a < home_h, "Cross-learned away rating should be smaller than direct home update"

    def test_pi_rating_diff(self):
        """pi_rating_diff should be home_h - away_a."""
        engineer = PiRatingFeatureEngineer()
        df = engineer.create_features({'matches': _make_matches()})
        for _, row in df.iterrows():
            expected = row['home_pi_rating_h'] - row['away_pi_rating_a']
            assert abs(row['pi_rating_diff'] - expected) < 1e-3  # Rounding tolerance

    def test_custom_parameters(self):
        """Engineer should accept custom lambda_, gamma, c parameters."""
        engineer = PiRatingFeatureEngineer(lambda_=0.05, gamma=0.50, c=2.0)
        assert engineer.lambda_ == 0.05
        assert engineer.gamma == 0.50
        assert engineer.c == 2.0
        # Should still produce output
        df = engineer.create_features({'matches': _make_matches(5)})
        assert len(df) == 5

    def test_feature_names(self):
        """get_feature_names should return all output feature names."""
        engineer = PiRatingFeatureEngineer()
        names = engineer.get_feature_names()
        assert 'home_pi_rating_h' in names
        assert 'pi_rating_diff' in names
        assert len(names) >= 6
