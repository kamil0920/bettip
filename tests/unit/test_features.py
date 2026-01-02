"""Unit tests for features module."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.features.cleaners import BasicDataCleaner, MatchDataCleaner, PlayerStatsDataCleaner
from src.features.engineers import (
    TeamFormFeatureEngineer,
    MatchOutcomeFeatureEngineer,
    HeadToHeadFeatureEngineer
)
from src.features.merger import DataMerger


class TestBasicDataCleaner:
    """Tests for BasicDataCleaner."""

    def test_remove_duplicates(self):
        """Test that duplicates are removed."""
        df = pd.DataFrame({
            'a': [1, 1, 2, 3],
            'b': ['x', 'x', 'y', 'z']
        })
        cleaner = BasicDataCleaner()
        result = cleaner.clean(df)

        assert len(result) == 3
        assert result['a'].tolist() == [1, 2, 3]

    def test_no_duplicates(self):
        """Test handling DataFrame without duplicates."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        cleaner = BasicDataCleaner()
        result = cleaner.clean(df)

        assert len(result) == 3


class TestMatchDataCleaner:
    """Tests for MatchDataCleaner."""

    def test_removes_na_scores(self):
        """Test that rows with missing scores are removed."""
        df = pd.DataFrame({
            'fixture_id': [1, 2, 3],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'ft_home': [2, None, 1],
            'ft_away': [1, 1, None]
        })
        cleaner = MatchDataCleaner()
        result = cleaner.clean(df)

        assert len(result) == 1
        assert result.iloc[0]['fixture_id'] == 1

    def test_converts_date(self):
        """Test that date column is converted to datetime."""
        df = pd.DataFrame({
            'fixture_id': [1],
            'date': ['2024-01-01T15:00:00'],
            'ft_home': [2],
            'ft_away': [1]
        })
        cleaner = MatchDataCleaner()
        result = cleaner.clean(df)

        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_sorts_by_date(self):
        """Test that matches are sorted by date."""
        df = pd.DataFrame({
            'fixture_id': [1, 2, 3],
            'date': ['2024-01-03', '2024-01-01', '2024-01-02'],
            'ft_home': [2, 1, 0],
            'ft_away': [1, 0, 0]
        })
        cleaner = MatchDataCleaner()
        result = cleaner.clean(df)

        assert result.iloc[0]['fixture_id'] == 2
        assert result.iloc[1]['fixture_id'] == 3
        assert result.iloc[2]['fixture_id'] == 1

    def test_column_mapping_raw_api_format(self):
        """Test that raw API columns are mapped to clean names."""
        df = pd.DataFrame({
            'fixture.id': [1, 2],
            'fixture.date': ['2024-01-01', '2024-01-02'],
            'teams.home.id': [100, 200],
            'teams.home.name': ['Team A', 'Team B'],
            'teams.away.id': [200, 100],
            'teams.away.name': ['Team B', 'Team A'],
            'goals.home': [2, 1],
            'goals.away': [1, 0],
            'score.halftime.home': [1, 0],
            'score.halftime.away': [0, 0],
            'fixture.status.short': ['FT', 'FT']
        })
        cleaner = MatchDataCleaner()
        result = cleaner.clean(df)

        # Check column mapping
        assert 'fixture_id' in result.columns
        assert 'home_team_id' in result.columns
        assert 'away_team_id' in result.columns
        assert 'ft_home' in result.columns
        assert 'ft_away' in result.columns
        assert 'home_team_name' in result.columns
        assert 'away_team_name' in result.columns

        # Check convenience columns
        assert 'home_team' in result.columns
        assert 'away_team' in result.columns
        assert result.iloc[0]['home_team'] == 'Team A'

    def test_column_mapping_preserves_existing_clean_columns(self):
        """Test that already clean columns are not overwritten."""
        df = pd.DataFrame({
            'fixture_id': [1],
            'date': ['2024-01-01'],
            'home_team_id': [100],
            'away_team_id': [200],
            'ft_home': [2],
            'ft_away': [1]
        })
        cleaner = MatchDataCleaner()
        result = cleaner.clean(df)

        assert result.iloc[0]['fixture_id'] == 1
        assert result.iloc[0]['home_team_id'] == 100


class TestPlayerStatsDataCleaner:
    """Tests for PlayerStatsDataCleaner."""

    def test_fills_numeric_na_with_zero(self):
        """Test that NaN in numeric columns are filled with 0."""
        df = pd.DataFrame({
            'player_id': [1, 2],
            'minutes': [90, 45],
            'goals': [1, np.nan],
            'assists': [np.nan, 1]
        })
        cleaner = PlayerStatsDataCleaner()
        result = cleaner.clean(df)

        assert result['goals'].tolist() == [1.0, 0.0]
        assert result['assists'].tolist() == [0.0, 1.0]

    def test_removes_zero_minutes(self):
        """Test that players with 0 minutes are removed."""
        df = pd.DataFrame({
            'player_id': [1, 2, 3],
            'minutes': [90, 0, 45],
            'goals': [1, 0, 0]
        })
        cleaner = PlayerStatsDataCleaner()
        result = cleaner.clean(df)

        assert len(result) == 2
        assert 2 not in result['player_id'].values


class TestTeamFormFeatureEngineer:
    """Tests for TeamFormFeatureEngineer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = TeamFormFeatureEngineer(n_matches=3)

    def test_create_form_features(self):
        """Test creation of team form features."""
        matches = pd.DataFrame({
            'fixture_id': [1, 2, 3, 4],
            'date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15', '2024-01-22']),
            'home_team_id': [100, 200, 100, 200],
            'away_team_id': [200, 100, 300, 100],
            'ft_home': [2, 1, 3, 0],
            'ft_away': [1, 0, 0, 2]
        })
        data = {'matches': matches}

        result = self.engineer.create_features(data)

        assert len(result) == 4
        assert 'home_wins_last_n' in result.columns
        assert 'away_wins_last_n' in result.columns
        assert 'home_points_last_n' in result.columns

    def test_first_match_no_history(self):
        """Test that first match has zero form values."""
        matches = pd.DataFrame({
            'fixture_id': [1],
            'date': pd.to_datetime(['2024-01-01']),
            'home_team_id': [100],
            'away_team_id': [200],
            'ft_home': [2],
            'ft_away': [1]
        })
        data = {'matches': matches}

        result = self.engineer.create_features(data)

        assert result.iloc[0]['home_wins_last_n'] == 0
        assert result.iloc[0]['away_wins_last_n'] == 0
        assert result.iloc[0]['home_points_last_n'] == 0

    def test_form_calculation_correct(self):
        """Test that form is calculated correctly."""
        # Team 100: Win (3pts), Draw (1pt), Loss (0pts) = 4 points
        matches = pd.DataFrame({
            'fixture_id': [1, 2, 3, 4],
            'date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15', '2024-01-22']),
            'home_team_id': [100, 100, 100, 100],
            'away_team_id': [200, 300, 400, 500],
            'ft_home': [2, 1, 0, 1],
            'ft_away': [1, 1, 2, 0]
        })
        data = {'matches': matches}

        result = self.engineer.create_features(data)

        # Check last match (fixture_id=4) - team 100 has played 3 matches
        last_row = result[result['fixture_id'] == 4].iloc[0]
        assert last_row['home_wins_last_n'] == 1  # 1 win
        assert last_row['home_draws_last_n'] == 1  # 1 draw
        assert last_row['home_losses_last_n'] == 1  # 1 loss
        assert last_row['home_points_last_n'] == 4  # 3 + 1 + 0


class TestMatchOutcomeFeatureEngineer:
    """Tests for MatchOutcomeFeatureEngineer."""

    def test_create_outcome_features(self):
        """Test creation of outcome target variables."""
        matches = pd.DataFrame({
            'fixture_id': [1, 2, 3],
            'date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15']),
            'home_team': ['Team A', 'Team B', 'Team A'],
            'away_team': ['Team B', 'Team A', 'Team C'],
            'ft_home': [2, 1, 1],
            'ft_away': [1, 1, 3]
        })
        data = {'matches': matches}

        engineer = MatchOutcomeFeatureEngineer()
        result = engineer.create_features(data)

        assert len(result) == 3
        assert 'home_win' in result.columns
        assert 'draw' in result.columns
        assert 'away_win' in result.columns
        assert 'total_goals' in result.columns

        # Check first match: home win
        assert result.iloc[0]['home_win'] == 1
        assert result.iloc[0]['draw'] == 0
        assert result.iloc[0]['away_win'] == 0
        assert result.iloc[0]['total_goals'] == 3

        # Check second match: draw
        assert result.iloc[1]['home_win'] == 0
        assert result.iloc[1]['draw'] == 1
        assert result.iloc[1]['away_win'] == 0

        # Check third match: away win
        assert result.iloc[2]['home_win'] == 0
        assert result.iloc[2]['draw'] == 0
        assert result.iloc[2]['away_win'] == 1

    def test_goal_difference(self):
        """Test goal difference calculation."""
        matches = pd.DataFrame({
            'fixture_id': [1],
            'date': pd.to_datetime(['2024-01-01']),
            'home_team': ['Team A'],
            'away_team': ['Team B'],
            'ft_home': [3],
            'ft_away': [1]
        })
        data = {'matches': matches}

        engineer = MatchOutcomeFeatureEngineer()
        result = engineer.create_features(data)

        assert result.iloc[0]['goal_difference'] == 2


class TestHeadToHeadFeatureEngineer:
    """Tests for HeadToHeadFeatureEngineer."""

    def test_no_h2h_history(self):
        """Test when teams have no head-to-head history."""
        matches = pd.DataFrame({
            'fixture_id': [1],
            'date': pd.to_datetime(['2024-01-01']),
            'home_team_id': [100],
            'away_team_id': [200],
            'ft_home': [2],
            'ft_away': [1]
        })
        data = {'matches': matches}

        engineer = HeadToHeadFeatureEngineer(n_h2h=3)
        result = engineer.create_features(data)

        assert result.iloc[0]['h2h_home_wins'] == 0
        assert result.iloc[0]['h2h_draws'] == 0
        assert result.iloc[0]['h2h_away_wins'] == 0
        assert result.iloc[0]['h2h_avg_goals'] == 0

    def test_h2h_with_history(self):
        """Test H2H calculation with previous matches."""
        matches = pd.DataFrame({
            'fixture_id': [1, 2, 3],
            'date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15']),
            'home_team_id': [100, 200, 100],
            'away_team_id': [200, 100, 200],
            'ft_home': [2, 1, 0],
            'ft_away': [1, 0, 0]  # Match 3: draw
        })
        data = {'matches': matches}

        engineer = HeadToHeadFeatureEngineer(n_h2h=3)
        result = engineer.create_features(data)

        # Check last match (fixture_id=3)
        # H2H: Match 1 - Team 100 won (home), Match 2 - Team 100 lost (away, 200 won at home)
        last_row = result[result['fixture_id'] == 3].iloc[0]
        assert last_row['h2h_home_wins'] == 1  # Team 100 won Match 1
        assert last_row['h2h_draws'] == 0
        assert last_row['h2h_away_wins'] == 1  # Team 200 won Match 2 (when 100 was away)
        assert last_row['h2h_avg_goals'] == 2.0  # (3 + 1) / 2


class TestDataMerger:
    """Tests for DataMerger."""

    def test_merge_single_feature_df(self):
        """Test merging single feature DataFrame."""
        base_df = pd.DataFrame({
            'fixture_id': [1, 2, 3],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        feature_df = pd.DataFrame({
            'fixture_id': [1, 2, 3],
            'feature_a': [10, 20, 30]
        })

        merger = DataMerger()
        result = merger.merge_all_features(base_df, [feature_df])

        assert len(result) == 3
        assert 'feature_a' in result.columns
        assert result['feature_a'].tolist() == [10, 20, 30]

    def test_merge_multiple_feature_dfs(self):
        """Test merging multiple feature DataFrames."""
        base_df = pd.DataFrame({
            'fixture_id': [1, 2, 3],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        feature_df1 = pd.DataFrame({
            'fixture_id': [1, 2, 3],
            'feature_a': [10, 20, 30]
        })
        feature_df2 = pd.DataFrame({
            'fixture_id': [1, 2, 3],
            'feature_b': [100, 200, 300]
        })

        merger = DataMerger()
        result = merger.merge_all_features(base_df, [feature_df1, feature_df2])

        assert len(result) == 3
        assert 'feature_a' in result.columns
        assert 'feature_b' in result.columns

    def test_merge_with_missing_keys(self):
        """Test merging when some keys are missing (left join)."""
        base_df = pd.DataFrame({
            'fixture_id': [1, 2, 3],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        feature_df = pd.DataFrame({
            'fixture_id': [1, 3],  # Missing fixture_id=2
            'feature_a': [10, 30]
        })

        merger = DataMerger()
        result = merger.merge_all_features(base_df, [feature_df])

        assert len(result) == 3
        assert pd.isna(result.iloc[1]['feature_a'])  # fixture_id=2 should be NaN
