"""Tests for odds feature engineering."""

import pandas as pd
import numpy as np
import pytest

from src.odds.odds_features import OddsFeatureEngineer


@pytest.fixture
def sample_odds_data():
    """Create sample odds data for testing."""
    return pd.DataFrame({
        'b365_home_open': [2.00, 1.50, 3.00, 2.20],
        'b365_draw_open': [3.50, 4.00, 3.30, 3.40],
        'b365_away_open': [3.80, 5.50, 2.40, 3.20],
        'b365_home_close': [1.90, 1.55, 2.80, 2.30],  # Home shortened
        'b365_draw_close': [3.60, 3.90, 3.40, 3.35],
        'b365_away_close': [4.00, 5.20, 2.50, 3.10],  # Away drifted
        'b365_over25': [1.90, 1.80, 2.10, 1.95],
        'b365_under25': [1.95, 2.05, 1.80, 1.90],
    })


@pytest.fixture
def engineer():
    """Create OddsFeatureEngineer instance."""
    return OddsFeatureEngineer()


class TestOddsFeatureEngineer:
    """Tests for OddsFeatureEngineer class."""

    def test_creates_probability_features(self, engineer, sample_odds_data):
        """Should create normalized probability features."""
        result = engineer.create_features(sample_odds_data)

        assert 'odds_home_prob' in result.columns
        assert 'odds_draw_prob' in result.columns
        assert 'odds_away_prob' in result.columns

        # Probabilities should sum to 1 (within floating point tolerance)
        prob_sum = result['odds_home_prob'] + result['odds_draw_prob'] + result['odds_away_prob']
        assert all(np.isclose(prob_sum, 1.0))

    def test_creates_movement_features(self, engineer, sample_odds_data):
        """Should create basic movement features."""
        result = engineer.create_features(sample_odds_data)

        assert 'odds_move_home' in result.columns
        assert 'odds_move_away' in result.columns
        assert 'odds_move_draw' in result.columns

        # First row: home shortened from 2.00 to 1.90
        assert result.iloc[0]['odds_move_home'] == pytest.approx(0.10)


class TestEnhancedLineMovement:
    """Tests for enhanced line movement features."""

    def test_line_movement_magnitude(self, engineer, sample_odds_data):
        """Should calculate total movement magnitude."""
        result = engineer.create_features(sample_odds_data)

        assert 'line_movement_magnitude' in result.columns
        # Magnitude should be non-negative
        assert all(result['line_movement_magnitude'] >= 0)

    def test_movement_consistency(self, engineer, sample_odds_data):
        """Should detect consistent vs inconsistent movement."""
        result = engineer.create_features(sample_odds_data)

        assert 'movement_consistent' in result.columns
        # Values should be 0 or 1
        assert set(result['movement_consistent'].unique()).issubset({0, 1})

    def test_sharp_money_direction(self, engineer, sample_odds_data):
        """Should indicate sharp money direction."""
        result = engineer.create_features(sample_odds_data)

        assert 'sharp_money_direction' in result.columns
        # Values should be -1, 0, or 1
        assert set(result['sharp_money_direction'].unique()).issubset({-1, 0, 1})

    def test_overround_change(self, engineer, sample_odds_data):
        """Should calculate overround change."""
        result = engineer.create_features(sample_odds_data)

        assert 'overround_change' in result.columns
        # Overround change is typically small
        assert all(result['overround_change'].abs() < 0.1)

    def test_movement_tiers(self, engineer, sample_odds_data):
        """Should categorize movement into tiers."""
        result = engineer.create_features(sample_odds_data)

        assert 'movement_tier_home' in result.columns
        assert 'movement_tier_away' in result.columns
        # Tiers should be 0, 1, 2, or 3
        assert all(result['movement_tier_home'].isin([0, 1, 2, 3]))
        assert all(result['movement_tier_away'].isin([0, 1, 2, 3]))

    def test_big_mover_flags(self, engineer, sample_odds_data):
        """Should flag large movements (>10%)."""
        result = engineer.create_features(sample_odds_data)

        assert 'big_mover_home' in result.columns
        assert 'big_mover_away' in result.columns
        # Should be binary
        assert set(result['big_mover_home'].unique()).issubset({0, 1})

    def test_favorite_drifting(self, engineer, sample_odds_data):
        """Should detect when favorite is drifting."""
        result = engineer.create_features(sample_odds_data)

        assert 'favorite_drifting' in result.columns
        # Should be binary
        assert set(result['favorite_drifting'].unique()).issubset({0, 1})

    def test_draw_steam(self, engineer, sample_odds_data):
        """Should detect steam moves on draw."""
        result = engineer.create_features(sample_odds_data)

        assert 'draw_steam' in result.columns
        assert 'odds_move_draw_pct' in result.columns

    def test_sharp_confidence(self, engineer, sample_odds_data):
        """Should calculate combined sharp confidence score."""
        result = engineer.create_features(sample_odds_data)

        assert 'sharp_confidence' in result.columns
        # Score ranges from 0 to 3 (sum of 3 indicators)
        assert all(result['sharp_confidence'] >= 0)
        assert all(result['sharp_confidence'] <= 3)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_opening_odds(self, engineer):
        """Should handle missing opening odds gracefully."""
        df = pd.DataFrame({
            'b365_home_close': [1.90, 1.55],
            'b365_draw_close': [3.60, 3.90],
            'b365_away_close': [4.00, 5.20],
        })
        result = engineer.create_features(df)

        # Movement features should not be created
        assert 'odds_move_home' not in result.columns
        # But probability features should still work
        assert 'odds_home_prob' in result.columns

    def test_missing_closing_odds(self, engineer):
        """Should handle missing closing odds by using opening odds."""
        # When use_closing_odds=True (default), it falls back to opening
        df = pd.DataFrame({
            'avg_home_open': [2.00, 1.50],
            'avg_draw_open': [3.50, 4.00],
            'avg_away_open': [3.80, 5.50],
        })
        result = engineer.create_features(df)

        # Should use opening odds as fallback for probabilities
        assert 'odds_home_prob' in result.columns
        # But movement features won't be available (need both open and close)
        assert 'line_movement_magnitude' not in result.columns

    def test_extreme_odds_movement(self, engineer):
        """Should handle extreme odds movements."""
        df = pd.DataFrame({
            'b365_home_open': [10.00],  # Long shot
            'b365_draw_open': [5.00],
            'b365_away_open': [1.20],  # Heavy favorite
            'b365_home_close': [5.00],  # 50% drop - huge steam
            'b365_draw_close': [5.50],
            'b365_away_close': [1.50],  # Drifted
        })
        result = engineer.create_features(df)

        # Big mover should be flagged
        assert result.iloc[0]['big_mover_home'] == 1
        # Movement tier should be highest
        assert result.iloc[0]['movement_tier_home'] == 3

    def test_no_movement(self, engineer):
        """Should handle no odds movement."""
        df = pd.DataFrame({
            'b365_home_open': [2.00],
            'b365_draw_open': [3.50],
            'b365_away_open': [3.80],
            'b365_home_close': [2.00],  # No change
            'b365_draw_close': [3.50],
            'b365_away_close': [3.80],
        })
        result = engineer.create_features(df)

        assert result.iloc[0]['odds_move_home'] == 0
        assert result.iloc[0]['line_movement_magnitude'] == 0
        assert result.iloc[0]['movement_tier_home'] == 0


class TestFeatureNames:
    """Tests for feature name listing."""

    def test_get_feature_names_includes_new_features(self, engineer):
        """Should list all enhanced features."""
        names = engineer.get_feature_names()

        enhanced_features = [
            'line_movement_magnitude',
            'movement_consistent',
            'sharp_money_direction',
            'overround_change',
            'movement_tier_home',
            'movement_tier_away',
            'big_mover_home',
            'big_mover_away',
            'favorite_drifting',
            'odds_move_draw_pct',
            'draw_steam',
            'sharp_confidence',
        ]

        for feat in enhanced_features:
            assert feat in names, f"{feat} not in feature names"
