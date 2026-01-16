"""Tests for match importance feature engineering."""

import pandas as pd
import numpy as np
import pytest

from src.features.match_importance import (
    MatchImportanceEngineer,
    add_derby_by_name,
    get_derby_teams_for_league,
    DERBY_NAMES,
    ALL_DERBIES,
    PREMIER_LEAGUE_DERBIES,
)


@pytest.fixture
def sample_match_data():
    """Create sample match data for testing."""
    return pd.DataFrame({
        'home_team_id': [40, 42, 33, 497, 100],  # Liverpool, Arsenal, Man Utd, Roma, Random
        'away_team_id': [45, 47, 50, 487, 101],  # Everton, Tottenham, Man City, Lazio, Random
        'home_team_name': ['Liverpool', 'Arsenal', 'Manchester United', 'Roma', 'Team A'],
        'away_team_name': ['Everton', 'Tottenham', 'Manchester City', 'Lazio', 'Team B'],
        'home_league_position': [1, 4, 15, 5, 10],
        'away_league_position': [13, 5, 2, 6, 11],
    })


@pytest.fixture
def relegation_data():
    """Create data for relegation battle testing."""
    return pd.DataFrame({
        'home_team_id': [100, 101, 102],
        'away_team_id': [103, 104, 105],
        'home_league_position': [18, 19, 1],  # Relegation zone
        'away_league_position': [20, 17, 20],  # One in zone, one out
    })


@pytest.fixture
def engineer():
    """Create MatchImportanceEngineer instance."""
    return MatchImportanceEngineer()


class TestDerbyDetection:
    """Tests for derby/rivalry detection."""

    def test_detects_merseyside_derby(self, engineer, sample_match_data):
        """Should detect Liverpool vs Everton as derby."""
        result = engineer.create_features(sample_match_data)
        # First row is Liverpool vs Everton
        assert result.iloc[0]['is_derby'] == 1

    def test_detects_north_london_derby(self, engineer, sample_match_data):
        """Should detect Arsenal vs Tottenham as derby."""
        result = engineer.create_features(sample_match_data)
        # Second row is Arsenal vs Tottenham
        assert result.iloc[1]['is_derby'] == 1

    def test_detects_manchester_derby(self, engineer, sample_match_data):
        """Should detect Man Utd vs Man City as derby."""
        result = engineer.create_features(sample_match_data)
        # Third row is Man Utd vs Man City
        assert result.iloc[2]['is_derby'] == 1

    def test_detects_rome_derby(self, engineer, sample_match_data):
        """Should detect Roma vs Lazio as derby."""
        result = engineer.create_features(sample_match_data)
        # Fourth row is Roma vs Lazio
        assert result.iloc[3]['is_derby'] == 1

    def test_non_derby_detected(self, engineer, sample_match_data):
        """Should not flag non-derbies."""
        result = engineer.create_features(sample_match_data)
        # Fifth row is random teams
        assert result.iloc[4]['is_derby'] == 0

    def test_derby_bidirectional(self, engineer):
        """Derby detection should work regardless of home/away."""
        df = pd.DataFrame({
            'home_team_id': [45, 40],  # Everton, Liverpool
            'away_team_id': [40, 45],  # Liverpool, Everton
            'home_league_position': [10, 5],
            'away_league_position': [5, 10],
        })
        result = engineer.create_features(df)
        assert result['is_derby'].sum() == 2


class TestPositionBasedFeatures:
    """Tests for position-based features."""

    def test_relegation_battle_detected(self, engineer, relegation_data):
        """Should detect relegation battle."""
        result = engineer.create_features(relegation_data)
        # First match: both teams in relegation zone (18, 20)
        assert result.iloc[0]['relegation_battle'] == 1
        # Third match: position 1 vs 20 - not a battle
        assert result.iloc[2]['relegation_battle'] == 0

    def test_title_race_detected(self, engineer):
        """Should detect title race matches."""
        df = pd.DataFrame({
            'home_team_id': [1, 2],
            'away_team_id': [3, 4],
            'home_league_position': [1, 1],
            'away_league_position': [2, 10],  # Title race, not title race
        })
        result = engineer.create_features(df)
        assert result.iloc[0]['title_race'] == 1
        assert result.iloc[1]['title_race'] == 0

    def test_european_race_detected(self, engineer):
        """Should detect European race (but not title race)."""
        df = pd.DataFrame({
            'home_team_id': [1, 2, 3],
            'away_team_id': [4, 5, 6],
            'home_league_position': [4, 1, 8],
            'away_league_position': [5, 2, 10],
        })
        result = engineer.create_features(df)
        # 4 vs 5 is European race
        assert result.iloc[0]['european_race'] == 1
        # 1 vs 2 is title race, not European race
        assert result.iloc[1]['european_race'] == 0
        # 8 vs 10 is neither
        assert result.iloc[2]['european_race'] == 0

    def test_position_gap_calculated(self, engineer, sample_match_data):
        """Should calculate position gap correctly."""
        result = engineer.create_features(sample_match_data)
        # First match: position 1 vs 13, gap = 12
        assert result.iloc[0]['position_gap'] == 12
        # Second match: position 4 vs 5, gap = 1
        assert result.iloc[1]['position_gap'] == 1

    def test_giant_killer_potential(self, engineer):
        """Should flag giant killer potential correctly."""
        df = pd.DataFrame({
            'home_team_id': [1, 2],
            'away_team_id': [3, 4],
            'home_league_position': [1, 10],
            'away_league_position': [15, 1],  # Away underdog with big gap, away favorite
        })
        result = engineer.create_features(df)
        # Position 1 vs 15 with away being lower = giant killer
        assert result.iloc[0]['giant_killer_potential'] == 1
        # Position 10 vs 1 with away being higher = not giant killer
        assert result.iloc[1]['giant_killer_potential'] == 0


class TestSixPointer:
    """Tests for six-pointer detection."""

    def test_six_pointer_relegation(self, engineer):
        """Six-pointer in relegation battle."""
        df = pd.DataFrame({
            'home_team_id': [1],
            'away_team_id': [2],
            'home_league_position': [18],
            'away_league_position': [19],  # Close in relegation zone
        })
        result = engineer.create_features(df)
        assert result.iloc[0]['six_pointer'] == 1

    def test_six_pointer_title_race(self, engineer):
        """Six-pointer in title race."""
        df = pd.DataFrame({
            'home_team_id': [1],
            'away_team_id': [2],
            'home_league_position': [1],
            'away_league_position': [2],  # Close at top
        })
        result = engineer.create_features(df)
        assert result.iloc[0]['six_pointer'] == 1

    def test_not_six_pointer_far_apart(self, engineer):
        """Not six-pointer when positions far apart."""
        df = pd.DataFrame({
            'home_team_id': [1],
            'away_team_id': [2],
            'home_league_position': [1],
            'away_league_position': [10],  # Too far apart
        })
        result = engineer.create_features(df)
        assert result.iloc[0]['six_pointer'] == 0


class TestMatchImportanceScore:
    """Tests for combined importance score."""

    def test_importance_score_range(self, engineer, sample_match_data):
        """Score should be in valid range."""
        result = engineer.create_features(sample_match_data)
        assert all(result['match_importance_score'] >= 0)
        assert all(result['match_importance_score'] <= 5)

    def test_derby_increases_score(self, engineer):
        """Derby should increase importance score."""
        df = pd.DataFrame({
            'home_team_id': [40, 100],  # Liverpool, Random
            'away_team_id': [45, 101],  # Everton (derby), Random
            'home_league_position': [10, 10],
            'away_league_position': [10, 10],
        })
        result = engineer.create_features(df)
        # Derby match should have higher score
        assert result.iloc[0]['match_importance_score'] > result.iloc[1]['match_importance_score']


class TestNameBasedDerbyDetection:
    """Tests for name-based derby detection fallback."""

    def test_detects_derby_by_name(self):
        """Should detect derby using team names."""
        df = pd.DataFrame({
            'home_team_name': ['Liverpool FC', 'Arsenal'],
            'away_team_name': ['Everton', 'Chelsea FC'],
        })
        result = add_derby_by_name(df, DERBY_NAMES)
        assert result.iloc[0]['is_derby_by_name'] == 1
        assert result.iloc[1]['is_derby_by_name'] == 1

    def test_handles_partial_names(self):
        """Should handle partial name matches."""
        df = pd.DataFrame({
            'home_team_name': ['Real Madrid CF', 'FC Barcelona'],
            'away_team_name': ['FC Barcelona', 'Real Madrid'],
        })
        result = add_derby_by_name(df, DERBY_NAMES)
        # El Clasico both ways
        assert result['is_derby_by_name'].sum() == 2


class TestDerbyMappings:
    """Tests for derby mapping data."""

    def test_premier_league_derbies_exist(self):
        """Should have Premier League derbies defined."""
        assert len(PREMIER_LEAGUE_DERBIES) > 0
        # Manchester derby should exist
        assert 33 in PREMIER_LEAGUE_DERBIES  # Man Utd
        assert 50 in PREMIER_LEAGUE_DERBIES[33]  # Man City

    def test_all_derbies_combined(self):
        """Should combine all league derbies."""
        assert len(ALL_DERBIES) > 0
        # Should include teams from multiple leagues
        assert 40 in ALL_DERBIES  # Liverpool (EPL)
        assert 541 in ALL_DERBIES  # Real Madrid (La Liga)

    def test_get_derby_teams_for_league(self):
        """Should return league-specific derbies."""
        epl_derbies = get_derby_teams_for_league('premier_league')
        assert len(epl_derbies) > 0
        assert 40 in epl_derbies  # Liverpool

        la_liga_derbies = get_derby_teams_for_league('la_liga')
        assert 541 in la_liga_derbies  # Real Madrid


class TestEdgeCases:
    """Tests for edge cases."""

    def test_missing_team_ids(self, engineer):
        """Should handle missing team IDs."""
        df = pd.DataFrame({
            'home_league_position': [1, 2],
            'away_league_position': [3, 4],
        })
        result = engineer.create_features(df)
        # Should still create position-based features
        assert 'position_gap' in result.columns
        # Derby should default to 0
        assert all(result['is_derby'] == 0)

    def test_missing_positions(self, engineer):
        """Should handle missing position data."""
        df = pd.DataFrame({
            'home_team_id': [40, 42],
            'away_team_id': [45, 47],
        })
        result = engineer.create_features(df)
        # Should still detect derbies
        assert all(result['is_derby'] == 1)
        # Position features should be 0 or NaN
        assert 'relegation_battle' in result.columns

    def test_nan_positions(self, engineer):
        """Should handle NaN positions gracefully."""
        df = pd.DataFrame({
            'home_team_id': [40],
            'away_team_id': [45],
            'home_league_position': [np.nan],
            'away_league_position': [5],
        })
        result = engineer.create_features(df)
        # Should not crash
        assert len(result) == 1


class TestFeatureNames:
    """Tests for feature name listing."""

    def test_get_feature_names(self, engineer):
        """Should list all created features."""
        names = engineer.get_feature_names()
        expected = [
            'is_derby', 'relegation_battle', 'title_race',
            'european_race', 'mid_table_clash', 'position_gap',
            'away_underdog', 'giant_killer_potential',
            'six_pointer', 'match_importance_score',
        ]
        for feat in expected:
            assert feat in names
