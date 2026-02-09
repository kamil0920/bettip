"""Tests for Kelly criterion integration and vig removal."""
import pytest
import numpy as np

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.odds.odds_features import remove_vig_2way
from src.ml.bankroll_manager import BankrollManager, RiskConfig


class TestRemoveVig2Way:
    """Tests for the 2-way vig removal utility."""

    def test_fair_odds_no_vig(self):
        """With fair odds (no vig), probabilities should be 0.5/0.5."""
        p_a, p_b = remove_vig_2way(2.0, 2.0)
        assert abs(p_a - 0.5) < 1e-6
        assert abs(p_b - 0.5) < 1e-6

    def test_sum_to_one(self):
        """Fair probabilities should always sum to 1."""
        p_a, p_b = remove_vig_2way(1.90, 1.90)
        assert abs(p_a + p_b - 1.0) < 1e-6

    def test_removes_overround(self):
        """Vig-removed probs should be lower than raw implied probs."""
        # Typical bookmaker odds with ~5% overround
        over_odds = 1.90
        under_odds = 1.90
        p_over, p_under = remove_vig_2way(over_odds, under_odds)

        raw_over = 1 / over_odds  # 0.5263
        assert p_over < raw_over
        assert abs(p_over - 0.5) < 1e-6  # Should be exactly 0.5

    def test_asymmetric_odds(self):
        """Test with asymmetric favorite/underdog odds."""
        p_over, p_under = remove_vig_2way(1.50, 2.80)
        assert p_over > p_under
        assert abs(p_over + p_under - 1.0) < 1e-6

    def test_heavy_vig(self):
        """Heavy vig should still produce valid probabilities."""
        # ~10% overround
        p_a, p_b = remove_vig_2way(1.80, 1.80)
        assert abs(p_a + p_b - 1.0) < 1e-6
        assert abs(p_a - 0.5) < 1e-6

    def test_invalid_odds_returns_half(self):
        """Invalid odds (<= 1.0) should return 0.5/0.5."""
        p_a, p_b = remove_vig_2way(0.5, 1.90)
        assert p_a == 0.5
        assert p_b == 0.5

    def test_real_world_over_under(self):
        """Real-world over/under 2.5 odds."""
        # Over 2.5 @ 1.87, Under 2.5 @ 2.00
        p_over, p_under = remove_vig_2way(1.87, 2.00)
        assert 0.50 < p_over < 0.55  # Over is slight favorite
        assert 0.45 < p_under < 0.50
        assert abs(p_over + p_under - 1.0) < 1e-6


class TestCalculateEdgeVigRemoval:
    """Tests for edge calculation with vig removal in daily recommendations."""

    def test_edge_calculation_with_vig_removed(self):
        """Verify edge is calculated against vig-removed probability."""
        # Import the function
        from experiments.generate_daily_recommendations import calculate_edge

        # over25 market: model says 60%, odds imply ~52.6% raw, ~50% fair
        match_odds = {
            "totals_over_avg": 1.90,
            "totals_under_avg": 1.90,
            "h2h_home_avg": None,
            "h2h_away_avg": None,
            "h2h_draw_avg": None,
        }
        edge = calculate_edge(0.60, "over25", match_odds)
        # Fair prob = 0.5, so edge = 0.60 - 0.50 = 0.10
        assert abs(edge - 0.10) < 0.01

    def test_edge_without_complement_odds(self):
        """Without complement odds, falls back to raw implied."""
        from experiments.generate_daily_recommendations import calculate_edge

        match_odds = {
            "totals_over_avg": 1.90,
            "totals_under_avg": None,  # No complement
        }
        edge = calculate_edge(0.60, "over25", match_odds)
        # Raw implied = 1/1.90 ≈ 0.5263, edge = 0.60 - 0.5263 ≈ 0.0737
        assert 0.07 < edge < 0.08

    def test_edge_h2h_market_vig_removal(self):
        """H2H markets should use 3-way vig removal."""
        from experiments.generate_daily_recommendations import calculate_edge

        match_odds = {
            "h2h_home_avg": 2.10,
            "h2h_draw_avg": 3.40,
            "h2h_away_avg": 3.50,
        }
        edge = calculate_edge(0.55, "home_win", match_odds)
        # Should be less than 0.55 - 1/2.10 ≈ 0.55 - 0.476 = 0.074
        # because vig-removed prob < raw implied
        assert edge > 0.05


class TestKellyStakeSizing:
    """Tests for Kelly criterion stake sizing."""

    def test_kelly_positive_edge(self):
        """Positive edge should produce positive stake."""
        config = RiskConfig(kelly_fraction=0.25, max_stake_per_bet=0.05)
        mgr = BankrollManager(total_bankroll=1000.0, config=config)

        stake = mgr.calculate_stake(
            market="home_win", probability=0.55, odds=2.10, edge=0.07
        )
        assert stake > 0

    def test_kelly_no_edge(self):
        """Zero/negative edge should produce zero stake."""
        config = RiskConfig(kelly_fraction=0.25, min_edge=0.02)
        mgr = BankrollManager(total_bankroll=1000.0, config=config)

        stake = mgr.calculate_stake(
            market="home_win", probability=0.45, odds=2.10, edge=0.01
        )
        assert stake == 0.0

    def test_kelly_max_stake_cap(self):
        """Stake should never exceed max_stake_fraction."""
        config = RiskConfig(
            kelly_fraction=1.0,  # Full Kelly (aggressive)
            max_stake_per_bet=0.05,
        )
        mgr = BankrollManager(total_bankroll=1000.0, config=config)

        stake = mgr.calculate_stake(
            market="home_win", probability=0.90, odds=5.0, edge=0.70
        )
        assert stake <= 1000.0 * 0.05 + 0.01  # Max 5% of bankroll

    def test_kelly_fraction_reduces_stake(self):
        """Quarter Kelly should produce ~25% of full Kelly stake."""
        config_full = RiskConfig(kelly_fraction=1.0, max_stake_per_bet=1.0)
        config_quarter = RiskConfig(kelly_fraction=0.25, max_stake_per_bet=1.0)

        mgr_full = BankrollManager(total_bankroll=1000.0, config=config_full)
        mgr_quarter = BankrollManager(total_bankroll=1000.0, config=config_quarter)

        stake_full = mgr_full.calculate_stake(
            market="home_win", probability=0.55, odds=2.10, edge=0.07
        )
        stake_quarter = mgr_quarter.calculate_stake(
            market="home_win", probability=0.55, odds=2.10, edge=0.07
        )

        # Quarter Kelly should be approximately 25% of full Kelly
        # (not exact due to Sharpe dampening)
        assert stake_quarter < stake_full
        ratio = stake_quarter / stake_full if stake_full > 0 else 0
        assert 0.2 < ratio < 0.3

    def test_kelly_odds_below_one(self):
        """Odds <= 1 should return zero stake."""
        config = RiskConfig(kelly_fraction=0.25)
        mgr = BankrollManager(total_bankroll=1000.0, config=config)

        stake = mgr.calculate_stake(
            market="home_win", probability=0.90, odds=0.95, edge=0.50
        )
        assert stake == 0.0
