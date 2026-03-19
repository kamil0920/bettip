"""Tests for goal-based K-factor in Elo ratings.

Validates that:
1. k_goal_lambda=0.0 produces identical output to original
2. k_goal_lambda>0 amplifies big wins
3. Config manager roundtrip works
4. Different lambda = different params_hash
"""

import numpy as np
import pandas as pd
import pytest


def _make_matches(results):
    """Create minimal matches DataFrame from (home_goals, away_goals) tuples."""
    rows = []
    for i, (h, a) in enumerate(results):
        rows.append({
            "fixture_id": 1000 + i,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "home_team_id": 1,
            "away_team_id": 2,
            "ft_home": h,
            "ft_away": a,
        })
    return pd.DataFrame(rows)


class TestEloGoalKFactor:
    """Test goal-based K-factor scaling."""

    def test_lambda_zero_matches_original(self):
        """k_goal_lambda=0.0 should produce identical output to default."""
        from src.features.engineers.ratings import ELORatingFeatureEngineer

        matches = _make_matches([(2, 1), (0, 3), (1, 1), (4, 0)])

        eng_default = ELORatingFeatureEngineer(k_factor=32.0, k_goal_lambda=0.0)
        eng_original = ELORatingFeatureEngineer(k_factor=32.0)

        result_default = eng_default.create_features({"matches": matches})
        result_original = eng_original.create_features({"matches": matches})

        # Elo values should be identical
        for col in ["home_elo", "away_elo", "elo_diff"]:
            np.testing.assert_array_almost_equal(
                result_default[col].values,
                result_original[col].values,
                decimal=10,
                err_msg=f"{col} differs with lambda=0.0 vs default",
            )

    def test_lambda_positive_amplifies_big_wins(self):
        """With lambda>0, a 4-0 win should produce larger Elo delta than 1-0."""
        from src.features.engineers.ratings import ELORatingFeatureEngineer

        # Two scenarios: team 1 wins 1-0 vs team 1 wins 4-0
        matches_narrow = _make_matches([(1, 0)])
        matches_big = _make_matches([(4, 0)])

        eng = ELORatingFeatureEngineer(k_factor=32.0, k_goal_lambda=0.3)

        result_narrow = eng.create_features({"matches": matches_narrow})
        result_big = eng.create_features({"matches": matches_big})

        # Before the match, Elo should be identical (both start at 1500)
        assert result_narrow["home_elo"].iloc[0] == result_big["home_elo"].iloc[0]

        # After the match (need to re-create to see post-match ratings)
        # Instead, let's check a 2-match scenario
        matches_2 = _make_matches([(1, 0), (0, 0)])  # narrow win, then draw
        matches_2_big = _make_matches([(4, 0), (0, 0)])  # big win, then draw

        result_2 = eng.create_features({"matches": matches_2})
        result_2_big = eng.create_features({"matches": matches_2_big})

        # After 1st match, home_elo should be higher for big win
        # Check the 2nd match's home_elo (reflects 1st match update)
        elo_after_narrow = result_2["home_elo"].iloc[1]
        elo_after_big = result_2_big["home_elo"].iloc[1]

        assert elo_after_big > elo_after_narrow, \
            f"4-0 win should give higher Elo than 1-0: {elo_after_big} vs {elo_after_narrow}"

    def test_lambda_affects_venue_ratings(self):
        """Goal-based K should also affect venue-specific ratings."""
        from src.features.engineers.ratings import ELORatingFeatureEngineer

        matches = _make_matches([(5, 0), (0, 0)])

        eng_fixed = ELORatingFeatureEngineer(k_factor=32.0, k_goal_lambda=0.0)
        eng_goal = ELORatingFeatureEngineer(k_factor=32.0, k_goal_lambda=0.3)

        r_fixed = eng_fixed.create_features({"matches": matches})
        r_goal = eng_goal.create_features({"matches": matches})

        # Venue Elo should differ between fixed K and goal-based K
        assert r_fixed["home_venue_elo"].iloc[1] != r_goal["home_venue_elo"].iloc[1]


class TestConfigManagerRoundtrip:
    """Test that elo_k_goal_lambda roundtrips through config system."""

    def test_config_to_registry_params(self):
        """Config should produce correct registry params."""
        from src.features.config_manager import BetTypeFeatureConfig

        config = BetTypeFeatureConfig(bet_type="home_win", elo_k_goal_lambda=0.25)
        params = config.to_registry_params()

        assert params["elo"]["k_goal_lambda"] == 0.25

    def test_params_hash_changes_with_lambda(self):
        """Different lambda values should produce different hashes."""
        from src.features.config_manager import BetTypeFeatureConfig

        config_0 = BetTypeFeatureConfig(bet_type="test", elo_k_goal_lambda=0.0)
        config_03 = BetTypeFeatureConfig(bet_type="test", elo_k_goal_lambda=0.3)

        assert config_0.params_hash() != config_03.params_hash()

    def test_default_lambda_is_zero(self):
        """Default config should have lambda=0.0."""
        from src.features.config_manager import BetTypeFeatureConfig

        config = BetTypeFeatureConfig(bet_type="test")
        assert config.elo_k_goal_lambda == 0.0

    def test_search_space_exists(self):
        """elo_k_goal_lambda should be in PARAMETER_SEARCH_SPACES."""
        from src.features.config_manager import PARAMETER_SEARCH_SPACES

        assert "elo_k_goal_lambda" in PARAMETER_SEARCH_SPACES
        lo, hi, typ = PARAMETER_SEARCH_SPACES["elo_k_goal_lambda"]
        assert lo == 0.0
        assert hi == 0.5
        assert typ == "float"

    def test_yaml_roundtrip(self, tmp_path):
        """Config should survive save/load cycle."""
        from src.features.config_manager import BetTypeFeatureConfig

        config = BetTypeFeatureConfig(bet_type="test_rt", elo_k_goal_lambda=0.15)
        path = config.save(params_dir=tmp_path)

        loaded = BetTypeFeatureConfig.load(path)
        assert loaded.elo_k_goal_lambda == 0.15
