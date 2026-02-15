"""Tests for LeaguePriorAdjuster — league-aware niche market calibration."""

import math

import numpy as np
import pandas as pd
import pytest

from src.calibration.league_prior_adjuster import (
    LeaguePriorAdjuster,
    _logit,
    _sigmoid,
    adjust_for_league,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    LeaguePriorAdjuster.reset()
    yield
    LeaguePriorAdjuster.reset()


@pytest.fixture
def sample_parquet(tmp_path):
    """Create a minimal features parquet with known base rates."""
    np.random.seed(42)
    n = 1000
    leagues = ["league_high", "league_low", "league_mid"]
    rows = []
    for league in leagues:
        for _ in range(n):
            if league == "league_high":
                cards = np.random.choice([2, 3, 4, 5, 6], p=[0.2, 0.4, 0.15, 0.15, 0.1])
                fouls = np.random.normal(22, 3)
                shots = np.random.normal(25, 4)
            elif league == "league_low":
                cards = np.random.choice([3, 4, 5, 6, 7], p=[0.1, 0.15, 0.3, 0.25, 0.2])
                fouls = np.random.normal(30, 3)
                shots = np.random.normal(28, 4)
            else:
                cards = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
                fouls = np.random.normal(26, 3)
                shots = np.random.normal(26, 4)
            rows.append({"league": league, "total_cards": cards, "total_fouls": fouls,
                        "total_shots": shots, "total_corners": np.random.normal(10, 2)})
    df = pd.DataFrame(rows)
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    return str(path)


class TestLogitSigmoid:
    """Test logit/sigmoid math."""

    def test_logit_sigmoid_roundtrip(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(_sigmoid(_logit(p)) - p) < 1e-10

    def test_logit_at_half(self):
        assert abs(_logit(0.5)) < 1e-10

    def test_sigmoid_at_zero(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-10

    def test_logit_extreme_values(self):
        # Should clamp, not crash
        assert math.isfinite(_logit(0.0))
        assert math.isfinite(_logit(1.0))
        assert _logit(0.0) < -10
        assert _logit(1.0) > 10


class TestLeaguePriorAdjuster:
    """Test adjuster with controlled data."""

    def test_high_variance_market_adjusted(self, sample_parquet):
        adj = LeaguePriorAdjuster(sample_parquet)
        # cards_under_35: league_high has ~60% under rate, league_low has ~10%
        # That spread > 15pp, so adjustment should apply
        prob = 0.80
        result_high = adj.adjust(prob, "cards_under_35", "league_high")
        result_low = adj.adjust(prob, "cards_under_35", "league_low")
        # league_high has higher under rate → should push prob UP
        # league_low has lower under rate → should push prob DOWN
        assert result_high > result_low

    def test_low_variance_market_not_adjusted(self, sample_parquet):
        adj = LeaguePriorAdjuster(sample_parquet)
        prob = 0.80
        # Corners have low variance (~10.9pp in real data, <15pp threshold)
        # With our sample data, check if it's low variance
        result = adj.adjust(prob, "corners_over_105", "league_high")
        # If corners spread < 15pp in our sample, should return original
        if not adj._high_variance.get("corners_over_105", False):
            assert result == prob

    def test_unknown_league_returns_original(self, sample_parquet):
        adj = LeaguePriorAdjuster(sample_parquet)
        prob = 0.75
        result = adj.adjust(prob, "cards_under_35", "nonexistent_league")
        assert result == prob

    def test_unknown_market_returns_original(self, sample_parquet):
        adj = LeaguePriorAdjuster(sample_parquet)
        prob = 0.75
        result = adj.adjust(prob, "unknown_market", "league_high")
        assert result == prob

    def test_strength_zero_no_change(self, sample_parquet):
        adj = LeaguePriorAdjuster(sample_parquet)
        prob = 0.75
        result = adj.adjust(prob, "cards_under_35", "league_high", strength=0.0)
        assert abs(result - prob) < 1e-10

    def test_over_under_symmetry(self, sample_parquet):
        """OVER and UNDER adjustments should be complementary."""
        adj = LeaguePriorAdjuster(sample_parquet)
        prob = 0.70
        under_result = adj.adjust(prob, "cards_under_35", "league_high")
        over_result = adj.adjust(1 - prob, "cards_over_35", "league_high")
        # under_result + over_result should be close to 1.0
        # (not exact due to logit-space non-linearity with different probs)
        # But if we adjust same prob for complementary markets in same league:
        assert abs(under_result + over_result - 1.0) < 0.05

    def test_adjustment_direction_under(self, sample_parquet):
        """For UNDER markets, high-under-rate league should increase prob."""
        adj = LeaguePriorAdjuster(sample_parquet)
        # league_high has many low-card games → high under rate
        prob = 0.50
        result = adj.adjust(prob, "cards_under_35", "league_high")
        # Should push UP since league_high's under rate > overall
        if adj._high_variance.get("cards_under_35", False):
            league_rate = adj._league_rates["cards_under_35"]["league_high"]
            overall = adj._overall_rates["cards_under_35"]
            if league_rate > overall:
                assert result > prob
            else:
                assert result < prob

    def test_adjustment_direction_over(self, sample_parquet):
        """For OVER markets, high-over-rate league should increase prob."""
        adj = LeaguePriorAdjuster(sample_parquet)
        prob = 0.50
        result = adj.adjust(prob, "cards_over_35", "league_low")
        # league_low has many high-card games → high over rate
        if adj._high_variance.get("cards_over_35", False):
            league_rate = adj._league_rates["cards_over_35"]["league_low"]
            overall = adj._overall_rates["cards_over_35"]
            if league_rate > overall:
                assert result > prob
            else:
                assert result < prob


class TestGracefulFallback:
    """Test behavior when parquet is missing or malformed."""

    def test_missing_parquet(self, tmp_path):
        adj = LeaguePriorAdjuster(str(tmp_path / "nonexistent.parquet"))
        result = adj.adjust(0.80, "cards_under_35", "ligue_1")
        assert result == 0.80

    def test_parquet_without_league_column(self, tmp_path):
        df = pd.DataFrame({"total_cards": [3, 4, 5], "total_fouls": [20, 25, 30],
                          "total_shots": [22, 26, 30], "total_corners": [8, 10, 12]})
        path = tmp_path / "no_league.parquet"
        df.to_parquet(path)
        adj = LeaguePriorAdjuster(str(path))
        result = adj.adjust(0.80, "cards_under_35", "ligue_1")
        assert result == 0.80


class TestSingleton:
    """Test singleton pattern."""

    def test_get_instance_returns_same_object(self, sample_parquet):
        inst1 = LeaguePriorAdjuster.get_instance(sample_parquet)
        inst2 = LeaguePriorAdjuster.get_instance(sample_parquet)
        assert inst1 is inst2

    def test_reset_clears_instance(self, sample_parquet):
        inst1 = LeaguePriorAdjuster.get_instance(sample_parquet)
        LeaguePriorAdjuster.reset()
        inst2 = LeaguePriorAdjuster.get_instance(sample_parquet)
        assert inst1 is not inst2


class TestModuleLevelFunction:
    """Test adjust_for_league convenience function."""

    def test_adjust_for_league_uses_singleton(self, sample_parquet):
        # Pre-initialize singleton with sample data
        LeaguePriorAdjuster.get_instance(sample_parquet)
        result = adjust_for_league(0.80, "cards_under_35", "league_high")
        # Should not crash and should return a valid probability
        assert 0.0 < result < 1.0
