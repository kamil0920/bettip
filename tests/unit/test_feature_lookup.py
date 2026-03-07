"""Unit tests for FeatureLookup."""

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_lookup import FeatureLookup


@pytest.fixture
def lookup_with_data(tmp_path):
    """Create a FeatureLookup pre-loaded with synthetic data (no file I/O)."""
    lookup = FeatureLookup(features_file=tmp_path / "dummy.parquet")
    # Build synthetic feature DataFrame
    df = pd.DataFrame({
        'fixture_id': [1, 2, 3, 4],
        'date': pd.to_datetime(['2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22']),
        'home_team': ['Liverpool', 'Man City', 'Liverpool', 'Arsenal'],
        'away_team': ['Man City', 'Arsenal', 'Arsenal', 'Liverpool'],
        'home_elo': [1800, 1850, 1810, 1780],
        'away_elo': [1850, 1780, 1780, 1810],
        'home_avg_yellows': [1.5, 2.0, 1.6, 1.8],
        'away_avg_yellows': [2.0, 1.8, 1.8, 1.6],
        'home_cards_ema': [1.5, 2.0, 1.6, 1.8],
        'away_cards_ema': [2.0, 1.8, 1.8, 1.6],
        'home_shots_ema': [12.0, 14.0, 13.0, 11.0],
        'away_shots_ema': [10.0, 11.0, 11.0, 13.0],
        'home_corners_ema': [5.0, 6.0, 5.5, 4.5],
        'away_corners_ema': [4.5, 4.0, 4.5, 5.5],
        'home_fouls_committed_ema': [11.0, 12.0, 11.5, 10.0],
        'away_fouls_committed_ema': [12.0, 10.0, 10.0, 11.5],
        'home_score': [2, 1, 3, 0],
        'away_score': [1, 2, 0, 1],
    })
    lookup._features_df = df.sort_values('date')
    lookup._available_features = [
        c for c in df.columns
        if c not in ('fixture_id', 'date', 'home_team', 'away_team',
                     'home_score', 'away_score')
    ]
    lookup._build_team_index()
    return lookup


class TestSafeGet:
    def test_returns_value_when_present(self):
        assert FeatureLookup._safe_get({'x': 3.5}, 'x', 0.0) == 3.5

    def test_returns_default_when_missing(self):
        assert FeatureLookup._safe_get({}, 'x', 7.0) == 7.0

    def test_returns_default_for_nan(self):
        assert FeatureLookup._safe_get({'x': float('nan')}, 'x', 1.0) == 1.0

    def test_returns_default_for_none(self):
        assert FeatureLookup._safe_get({'x': None}, 'x', 2.0) == 2.0


class TestBuildTeamIndex:
    def test_index_has_home_and_away_entries(self, lookup_with_data):
        assert 'home_Liverpool' in lookup_with_data._team_features
        assert 'away_Man City' in lookup_with_data._team_features

    def test_latest_match_used(self, lookup_with_data):
        # Liverpool's last home game is fixture 3 (2026-01-15)
        features = lookup_with_data._team_features['home_Liverpool']
        assert features['home_elo'] == 1810


class TestFindTeamFeatures:
    def test_exact_match(self, lookup_with_data):
        result = lookup_with_data._find_team_features('Liverpool', is_home=True)
        assert result is not None
        assert result['home_elo'] == 1810

    def test_fuzzy_match_substring(self, lookup_with_data):
        # "Man City" should match via substring
        result = lookup_with_data._find_team_features('Man City', is_home=True)
        assert result is not None

    def test_no_match_returns_none(self, lookup_with_data):
        result = lookup_with_data._find_team_features('Nonexistent FC', is_home=True)
        assert result is None


class TestGetTeamFeatures:
    def test_returns_dataframe(self, lookup_with_data):
        result = lookup_with_data.get_team_features('Liverpool', 'Man City')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_combines_home_and_away(self, lookup_with_data):
        result = lookup_with_data.get_team_features('Liverpool', 'Man City')
        assert result is not None
        # home_elo should come from Liverpool's last home game
        assert 'home_elo' in result.columns
        # away_elo should come from Man City's last away game
        assert 'away_elo' in result.columns

    def test_returns_none_for_unknown_team(self, lookup_with_data):
        result = lookup_with_data.get_team_features('Unknown FC', 'Man City')
        assert result is None

    def test_respects_feature_list(self, lookup_with_data):
        result = lookup_with_data.get_team_features(
            'Liverpool', 'Man City',
            feature_list=['home_elo', 'away_elo'],
        )
        assert result is not None
        assert set(result.columns) == {'home_elo', 'away_elo'}


class TestRecomputeCrossFeatures:
    def test_yellows_product_recomputed(self, lookup_with_data):
        combined = {
            'home_avg_yellows': 2.0,
            'away_avg_yellows': 3.0,
            'cross_yellows_product': 999.0,  # stale value
            'cross_yellows_total': 999.0,
        }
        lookup_with_data._recompute_cross_features(combined)
        assert combined['cross_yellows_product'] == pytest.approx(6.0)
        assert combined['cross_yellows_total'] == pytest.approx(5.0)

    def test_shots_interactions(self, lookup_with_data):
        combined = {
            'home_shots_ema': 12.0,
            'away_shots_ema': 10.0,
            'cross_shots_product': 0.0,
            'cross_shots_total': 0.0,
            'cross_shots_diff': 0.0,
        }
        lookup_with_data._recompute_cross_features(combined)
        assert combined['cross_shots_product'] == pytest.approx(120.0)
        assert combined['cross_shots_total'] == pytest.approx(22.0)
        assert combined['cross_shots_diff'] == pytest.approx(2.0)

    def test_diff_features_recomputed(self, lookup_with_data):
        combined = {
            'home_elo': 1800.0,
            'away_elo': 1700.0,
            'elo_diff': 0.0,  # stale
        }
        lookup_with_data._recompute_cross_features(combined)
        assert combined['elo_diff'] == pytest.approx(100.0)

    def test_skips_absent_features(self, lookup_with_data):
        combined = {'home_avg_yellows': 2.0, 'away_avg_yellows': 3.0}
        # Should not crash if cross features are absent
        lookup_with_data._recompute_cross_features(combined)
        assert 'cross_yellows_product' not in combined


class TestGetH2HFeatures:
    def test_returns_h2h_stats(self, lookup_with_data):
        features = lookup_with_data.get_h2h_features('Liverpool', 'Man City')
        assert features['h2h_matches'] > 0
        assert 'h2h_home_wins' in features
        assert 'h2h_avg_goals' in features

    def test_returns_empty_for_no_history(self, lookup_with_data):
        features = lookup_with_data.get_h2h_features('Liverpool', 'Unknown FC')
        assert features == {}

    def test_h2h_counts_correct(self, lookup_with_data):
        # Only fixture 1 is Liverpool vs Man City
        features = lookup_with_data.get_h2h_features('Liverpool', 'Man City')
        assert features['h2h_matches'] == 1
        # Liverpool 2-1 Man City -> home win for Liverpool
        assert features['h2h_home_wins'] == 1

    def test_h2h_win_pct(self, lookup_with_data):
        features = lookup_with_data.get_h2h_features('Liverpool', 'Arsenal')
        # Fixtures 3 and 4 involve Liverpool vs Arsenal
        assert features['h2h_matches'] == 2
        assert 'h2h_home_win_pct' in features
