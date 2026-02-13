"""Integration tests for prematch intelligence pipeline."""
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class TestMorningPredictionInjectsRefereeFeatures:
    """Test that morning prediction flow injects referee features."""

    def test_morning_prediction_injects_referee_features(self, tmp_path):
        """Feature injector should add referee features from cache."""
        from src.ml.feature_injector import ExternalFeatureInjector

        # Create referee cache
        referee_cache = pd.DataFrame({
            'referee_name': ['Michael Oliver'],
            'matches': [50],
            'total_yellows': [200],
            'total_reds': [10],
            'total_fouls': [1100],
            'total_corners': [520],
            'home_wins': [23],
            'draws': [12],
            'away_wins': [15],
            'total_goals': [140],
        })
        ref_path = tmp_path / 'referee_stats.parquet'
        referee_cache.to_parquet(ref_path)

        injector = ExternalFeatureInjector(
            referee_cache_path=str(ref_path),
            enable_weather=False,
            enable_lineups=False,
        )

        features_df = pd.DataFrame({
            'fixture_id': [1],
            'home_elo': [1600],
            'away_elo': [1500],
        })

        # Morning prediction — referee known, no lineups
        result = injector.inject_features(features_df, {
            'referee': 'Michael Oliver',
        })

        # Referee features should be present
        assert result['ref_cards_avg'].iloc[0] == pytest.approx(4.2, rel=0.1)
        assert result['ref_fouls_avg'].iloc[0] == pytest.approx(22.0, rel=0.1)
        # Original features preserved
        assert result['home_elo'].iloc[0] == 1600


class TestPreKickoffPredictionInjectsLineupData:
    """Test that pre-kickoff flow injects lineup features."""

    def test_pre_kickoff_prediction_injects_lineup_data(self, tmp_path):
        """When lineups provided, injector should compute lineup features."""
        from src.ml.feature_injector import ExternalFeatureInjector

        # Player stats cache
        player_cache = pd.DataFrame({
            'player_id': [101, 102, 201, 202],
            'player_name': ['Alisson', 'Salah', 'Ederson', 'Haaland'],
            'avg_rating': [7.0, 7.8, 6.8, 8.2],
            'total_minutes': [2700, 2700, 2600, 2600],
            'matches_played': [30, 30, 29, 29],
            'goals_per_90': [0.0, 0.7, 0.0, 0.9],
            'assists_per_90': [0.0, 0.3, 0.0, 0.2],
            'position': ['G', 'F', 'G', 'F'],
        })
        player_path = tmp_path / 'player_stats.parquet'
        player_cache.to_parquet(player_path)

        # Team rosters cache
        roster_cache = pd.DataFrame({
            'team_name': ['Liverpool', 'Liverpool', 'Man City', 'Man City'],
            'player_id': [101, 102, 201, 202],
            'player_name': ['Alisson', 'Salah', 'Ederson', 'Haaland'],
            'starts_in_last_n': [10, 9, 10, 9],
            'avg_rating': [7.0, 7.8, 6.8, 8.2],
            'position': ['G', 'F', 'G', 'F'],
        })
        roster_path = tmp_path / 'team_rosters.parquet'
        roster_cache.to_parquet(roster_path)

        injector = ExternalFeatureInjector(
            referee_cache_path=str(tmp_path / 'empty.parquet'),
            player_stats_cache_path=str(player_path),
            team_rosters_cache_path=str(roster_path),
            enable_referee=False,
            enable_weather=False,
            enable_lineups=True,
        )

        features_df = pd.DataFrame({
            'fixture_id': [1],
            'home_xi_avg_rating': [6.5],  # Historical default
            'away_xi_avg_rating': [6.5],
        })

        # Pre-kickoff: lineups available
        result = injector.inject_features(features_df, {
            'home_lineup': {'starting_xi': [{'id': 101}, {'id': 102}]},
            'away_lineup': {'starting_xi': [{'id': 201}, {'id': 202}]},
            'home_team': 'Liverpool',
            'away_team': 'Man City',
        })

        # Lineup features should override historical defaults
        assert result['home_xi_avg_rating'].iloc[0] == pytest.approx(7.4, rel=0.05)
        assert result['away_xi_avg_rating'].iloc[0] == pytest.approx(7.5, rel=0.05)

        # GK rating should be set
        assert result['home_gk_rating_avg'].iloc[0] == pytest.approx(7.0, rel=0.01)

        # Missing rating should be 0 (all expected starters present)
        assert result['home_missing_rating'].iloc[0] == pytest.approx(0.0, abs=0.01)
        assert result['away_missing_rating'].iloc[0] == pytest.approx(0.0, abs=0.01)

        # xi_rating_advantage alias
        assert 'xi_rating_advantage' in result.columns

    def test_morning_without_lineups_keeps_historical(self, tmp_path):
        """Morning run (no lineups) should keep FeatureLookup historical values."""
        from src.ml.feature_injector import ExternalFeatureInjector

        player_cache = pd.DataFrame({
            'player_id': [101], 'player_name': ['Salah'],
            'avg_rating': [7.8], 'total_minutes': [2700],
            'matches_played': [30], 'goals_per_90': [0.7],
            'assists_per_90': [0.3], 'position': ['F'],
        })
        player_path = tmp_path / 'player_stats.parquet'
        player_cache.to_parquet(player_path)

        injector = ExternalFeatureInjector(
            referee_cache_path=str(tmp_path / 'empty.parquet'),
            player_stats_cache_path=str(player_path),
            enable_referee=False,
            enable_weather=False,
            enable_lineups=True,
        )

        features_df = pd.DataFrame({
            'fixture_id': [1],
            'home_xi_avg_rating': [7.2],  # Historical value from FeatureLookup
            'away_xi_avg_rating': [6.9],
        })

        # Morning: no lineups
        result = injector.inject_features(features_df, {})

        # Historical values should be preserved
        assert result['home_xi_avg_rating'].iloc[0] == 7.2
        assert result['away_xi_avg_rating'].iloc[0] == 6.9


class TestFeatureNamesAlignWithDeploymentConfig:
    """Test that injected feature names match what deployed models expect."""

    def test_feature_names_align_with_deployment_config(self):
        """All lineup features in deployment config should be producible by injector."""
        config_path = project_root / "config" / "sniper_deployment.json"
        if not config_path.exists():
            pytest.skip("Deployment config not found")

        with open(config_path) as f:
            config = json.load(f)

        # Collect all lineup-related features from all markets
        lineup_features_in_config = set()
        lineup_prefixes = (
            'home_xi_', 'away_xi_', 'xi_rating_', 'lineup_',
            'home_missing_', 'away_missing_', 'missing_rating_',
            'home_gk_', 'away_gk_',
        )

        for market_name, market_config in config.get("markets", {}).items():
            for feat in market_config.get("selected_features", []):
                if any(feat.startswith(p) for p in lineup_prefixes):
                    lineup_features_in_config.add(feat)

        # These are features the injector CAN produce from lineups
        injectable_features = {
            'home_xi_avg_rating', 'away_xi_avg_rating',
            'xi_rating_advantage', 'lineup_rating_diff',
            'lineup_offensive_diff',
            'home_xi_goals_per_90', 'away_xi_goals_per_90',
            'home_xi_assists_per_90', 'away_xi_assists_per_90',
            'home_gk_rating_avg', 'away_gk_rating_avg',
            'home_missing_rating', 'away_missing_rating',
            'missing_rating_disadvantage',
        }

        # Walk-forward features that CANNOT be computed from single lineup
        walkforward_features = {
            'home_lineup_stability', 'away_lineup_stability',
            'lineup_stability_diff', 'stars_advantage',
            'home_key_players_missing',
            'home_gk_experience', 'away_gk_experience',
        }

        # Every injectable feature in config should be in our injectable set
        for feat in lineup_features_in_config:
            if feat in walkforward_features:
                continue  # These are kept from FeatureLookup
            assert feat in injectable_features, (
                f"Feature '{feat}' found in deployment config but not in "
                f"injector's producible features"
            )
