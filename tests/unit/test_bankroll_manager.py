"""
Unit tests for BankrollManager.

Tests edge cases:
- Daily limit enforcement (max bets, stop-loss, take-profit)
- Sharpe-dampened stake calculation
- Market allocation by Sharpe
- Zero/negative Sharpe handling
- Minimum edge cutoff
"""

import pytest
from datetime import date
from unittest.mock import patch, MagicMock

from src.ml.bankroll_manager import (
    BankrollManager,
    RiskConfig,
    MarketPerformance,
    DailyState,
    create_bankroll_manager,
)


class TestDailyLimitEnforcement:
    """Test daily betting limits."""

    def test_max_daily_bets_enforced(self):
        """Should block bets after max daily bets reached."""
        config = RiskConfig(max_daily_bets=3)
        manager = BankrollManager(total_bankroll=1000, config=config)

        # First 3 bets should be allowed
        for i in range(3):
            can_bet, reason = manager.can_bet_today("away_win")
            assert can_bet, f"Bet {i+1} should be allowed"
            manager.record_bet("away_win", 50)

        # 4th bet should be blocked
        can_bet, reason = manager.can_bet_today("away_win")
        assert not can_bet
        assert "limit reached" in reason.lower()

    def test_stop_loss_triggered(self):
        """Should block bets after stop-loss threshold hit."""
        config = RiskConfig(
            max_daily_bets=10,
            stop_loss_daily=0.10,  # 10% stop loss
        )
        manager = BankrollManager(total_bankroll=1000, config=config)

        # Simulate losses
        manager.record_bet("away_win", 50, pnl=-50)
        manager.record_bet("away_win", 50, pnl=-51)  # Total loss: 101 (10.1%)

        can_bet, reason = manager.can_bet_today("away_win")
        assert not can_bet
        assert "stop-loss" in reason.lower()

    def test_take_profit_reached(self):
        """Should block bets after take-profit threshold hit."""
        config = RiskConfig(
            max_daily_bets=10,
            take_profit_daily=0.20,  # 20% take profit
        )
        manager = BankrollManager(total_bankroll=1000, config=config)

        # Simulate big win
        manager.record_bet("away_win", 50, pnl=201)  # 20.1% profit

        can_bet, reason = manager.can_bet_today("away_win")
        assert not can_bet
        assert "take-profit" in reason.lower()

    def test_daily_state_resets_on_new_day(self):
        """Daily counters should reset on new day."""
        config = RiskConfig()  # Explicit config to avoid YAML loading
        manager = BankrollManager(total_bankroll=1000, config=config)
        manager.record_bet("away_win", 50)
        manager.record_bet("away_win", 50)

        assert manager.daily_state.bets_placed == 2

        # Simulate new day
        manager.daily_state.date = date(2020, 1, 1)  # Old date
        manager.daily_state.reset_if_new_day()

        assert manager.daily_state.bets_placed == 0
        assert manager.daily_state.date == date.today()


class TestSharpeDampenedStaking:
    """Test Sharpe-dampened Kelly stake calculation."""

    def test_basic_stake_calculation(self):
        """Should calculate stake using Sharpe dampening formula."""
        config = RiskConfig(
            kelly_fraction=0.25,
            sharpe_dampening_factor=15.0,
            min_edge=0.02,
        )
        manager = BankrollManager(total_bankroll=1000, config=config)

        # Add market with known Sharpe
        manager.market_performance["away_win"] = MarketPerformance(
            market="away_win",
            sharpe_ratio=3.0,
            expected_roi=0.15,
            precision=0.65,
            total_bets=100,
        )
        manager.allocate_by_sharpe()

        # Calculate stake
        stake = manager.calculate_stake(
            market="away_win",
            probability=0.60,
            odds=2.5,
            edge=0.10,
        )

        assert stake > 0
        # Stake should be less than max (5% of bankroll = 50)
        assert stake <= 50

    def test_edge_below_minimum_returns_zero(self):
        """Should return 0 stake if edge below minimum."""
        config = RiskConfig(min_edge=0.05)
        manager = BankrollManager(total_bankroll=1000, config=config)

        stake = manager.calculate_stake(
            market="away_win",
            probability=0.52,
            odds=2.0,
            edge=0.02,  # Below 5% minimum
        )

        assert stake == 0.0

    def test_negative_kelly_returns_zero(self):
        """Should return 0 for negative Kelly (expected loss)."""
        config = RiskConfig()  # Explicit config to avoid YAML loading
        manager = BankrollManager(total_bankroll=1000, config=config)

        stake = manager.calculate_stake(
            market="away_win",
            probability=0.30,  # Low probability
            odds=2.0,  # Implied 50%
            edge=-0.20,  # Negative edge
        )

        assert stake == 0.0

    def test_higher_sharpe_increases_stake(self):
        """Higher Sharpe should result in higher stake (all else equal)."""
        config = RiskConfig()  # Explicit config to avoid YAML loading
        manager = BankrollManager(total_bankroll=1000, config=config)

        # Low Sharpe market
        manager.market_performance["low_sharpe"] = MarketPerformance(
            market="low_sharpe",
            sharpe_ratio=1.0,
            expected_roi=0.05,
            precision=0.55,
            total_bets=100,
        )

        # High Sharpe market
        manager.market_performance["high_sharpe"] = MarketPerformance(
            market="high_sharpe",
            sharpe_ratio=5.0,
            expected_roi=0.20,
            precision=0.70,
            total_bets=100,
        )

        manager.allocate_by_sharpe()

        # Same bet parameters
        params = dict(probability=0.60, odds=2.5, edge=0.10)

        stake_low = manager.calculate_stake(market="low_sharpe", **params)
        stake_high = manager.calculate_stake(market="high_sharpe", **params)

        # High Sharpe should have higher allocation AND higher Sharpe factor
        assert stake_high > stake_low

    def test_stake_capped_at_max(self):
        """Stake should never exceed max_stake_per_bet."""
        config = RiskConfig(
            max_stake_per_bet=0.02,  # 2% max
            kelly_fraction=1.0,  # Full Kelly (aggressive)
            min_edge=0.01,
        )
        manager = BankrollManager(total_bankroll=1000, config=config)

        # Very favorable bet that would suggest large stake
        manager.market_performance["away_win"] = MarketPerformance(
            market="away_win",
            sharpe_ratio=10.0,
            expected_roi=0.50,
            precision=0.80,
            total_bets=100,
        )
        manager.allocate_by_sharpe()

        stake = manager.calculate_stake(
            market="away_win",
            probability=0.80,
            odds=3.0,
            edge=0.30,
        )

        # Should be capped at 2% = 20
        assert stake <= 20.0


class TestMarketAllocation:
    """Test market-level bankroll allocation."""

    def test_allocation_proportional_to_sharpe(self):
        """Markets should get allocation proportional to Sharpe ratio."""
        config = RiskConfig()  # Explicit config to avoid YAML loading
        manager = BankrollManager(total_bankroll=1000, config=config)

        # Two markets: one with 2x the Sharpe
        manager.market_performance["market_a"] = MarketPerformance(
            market="market_a",
            sharpe_ratio=2.0,
            expected_roi=0.10,
            precision=0.60,
            total_bets=100,
        )
        manager.market_performance["market_b"] = MarketPerformance(
            market="market_b",
            sharpe_ratio=4.0,  # 2x Sharpe
            expected_roi=0.20,
            precision=0.70,
            total_bets=100,
        )

        allocation = manager.allocate_by_sharpe()

        # market_b should get ~2x the allocation
        ratio = allocation["market_b"] / allocation["market_a"]
        assert 1.9 < ratio < 2.1

    def test_zero_sharpe_gets_no_allocation(self):
        """Markets with zero or negative Sharpe should get no allocation."""
        config = RiskConfig()  # Explicit config to avoid YAML loading
        manager = BankrollManager(total_bankroll=1000, config=config)

        manager.market_performance["profitable"] = MarketPerformance(
            market="profitable",
            sharpe_ratio=2.0,
            expected_roi=0.10,
            precision=0.60,
            total_bets=100,
        )
        manager.market_performance["unprofitable"] = MarketPerformance(
            market="unprofitable",
            sharpe_ratio=-0.5,
            expected_roi=-0.05,
            precision=0.45,
            total_bets=100,
        )

        allocation = manager.allocate_by_sharpe()

        assert "profitable" in allocation
        assert "unprofitable" not in allocation

    def test_single_market_gets_full_allocation(self):
        """Single profitable market should get full bankroll."""
        config = RiskConfig()  # Explicit config to avoid YAML loading
        manager = BankrollManager(total_bankroll=1000, config=config)

        manager.market_performance["only_market"] = MarketPerformance(
            market="only_market",
            sharpe_ratio=2.0,
            expected_roi=0.10,
            precision=0.60,
            total_bets=100,
        )

        allocation = manager.allocate_by_sharpe()

        assert allocation["only_market"] == 1000.0

    def test_no_profitable_markets_empty_allocation(self):
        """No allocation if no profitable markets."""
        config = RiskConfig()  # Explicit config to avoid YAML loading
        manager = BankrollManager(total_bankroll=1000, config=config)

        manager.market_performance["losing_a"] = MarketPerformance(
            market="losing_a",
            sharpe_ratio=-1.0,
            expected_roi=-0.10,
            precision=0.40,
            total_bets=100,
        )
        manager.market_performance["losing_b"] = MarketPerformance(
            market="losing_b",
            sharpe_ratio=0.0,
            expected_roi=0.0,
            precision=0.50,
            total_bets=100,
        )

        allocation = manager.allocate_by_sharpe()

        assert len(allocation) == 0


class TestDailySummary:
    """Test daily summary reporting."""

    def test_summary_includes_all_fields(self):
        """Summary should include all relevant fields."""
        config = RiskConfig()  # Explicit config to avoid YAML loading
        manager = BankrollManager(total_bankroll=1000, config=config)
        manager.record_bet("away_win", 50, pnl=25)
        manager.record_bet("btts", 30, pnl=-30)

        summary = manager.get_daily_summary()

        assert "date" in summary
        assert "bets_placed" in summary
        assert summary["bets_placed"] == 2
        assert "total_stake" in summary
        assert summary["total_stake"] == 80
        assert "total_pnl" in summary
        assert summary["total_pnl"] == -5
        assert "pnl_percentage" in summary
        assert "bets_by_market" in summary
        assert summary["bets_by_market"]["away_win"] == 1
        assert summary["bets_by_market"]["btts"] == 1


class TestUnprofitableMarketBlocking:
    """Test that unprofitable markets are blocked."""

    def test_unprofitable_market_blocked(self):
        """Should not allow bets on unprofitable markets."""
        config = RiskConfig()  # Explicit config to avoid YAML loading
        manager = BankrollManager(total_bankroll=1000, config=config)

        manager.market_performance["losing_market"] = MarketPerformance(
            market="losing_market",
            sharpe_ratio=-0.5,
            expected_roi=-0.10,
            precision=0.40,
            total_bets=100,
        )

        can_bet, reason = manager.can_bet_today("losing_market")

        assert not can_bet
        assert "not profitable" in reason.lower()


class TestFactoryFunction:
    """Test create_bankroll_manager factory."""

    def test_factory_creates_manager(self):
        """Factory should create configured manager."""
        # Use a non-existent path so it doesn't try to load YAML
        manager = create_bankroll_manager(
            bankroll=500,
            strategies_path="/nonexistent/path.yaml",
        )

        assert manager.total_bankroll == 500
        assert isinstance(manager, BankrollManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
