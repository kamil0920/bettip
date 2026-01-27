"""
Bankroll Management with Sharpe-Dampened Kelly Criterion.

Implements risk controls for betting operations:
- Sharpe-dampened Kelly staking
- Market-level bankroll allocation by Sharpe ratio
- Daily limit enforcement (bets, stop-loss, take-profit)
- Minimum edge requirements

This module enforces the risk_management settings from strategies.yaml
which were previously DEFINED but NOT ENFORCED.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class MarketPerformance:
    """Historical performance metrics for a market."""
    market: str
    sharpe_ratio: float
    expected_roi: float
    precision: float
    total_bets: int
    coefficient_of_variation: float = 0.0  # ROI standard deviation / mean

    @property
    def is_profitable(self) -> bool:
        """Check if market has positive expected value."""
        return self.expected_roi > 0 and self.sharpe_ratio > 0


@dataclass
class DailyState:
    """Tracks daily betting activity for limit enforcement."""
    date: date
    bets_placed: int = 0
    total_stake: float = 0.0
    total_pnl: float = 0.0
    bets_by_market: Dict[str, int] = field(default_factory=dict)

    def reset_if_new_day(self) -> None:
        """Reset counters if it's a new day."""
        if date.today() != self.date:
            self.date = date.today()
            self.bets_placed = 0
            self.total_stake = 0.0
            self.total_pnl = 0.0
            self.bets_by_market = {}


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_daily_bets: int = 10
    max_stake_per_bet: float = 0.05  # 5% of bankroll
    stop_loss_daily: float = 0.10  # Stop if 10% down
    take_profit_daily: float = 0.20  # Lock in if 20% up
    min_edge: float = 0.02  # 2% minimum edge to bet
    kelly_fraction: float = 0.25  # Quarter Kelly
    sharpe_dampening_factor: float = 15.0  # Divide Sharpe by this for dampening


class BankrollManager:
    """
    Manages bankroll allocation and stake sizing with risk controls.

    Key features:
    - Sharpe-dampened Kelly: Reduces stakes for high-Sharpe markets to
      account for estimation uncertainty
    - Market allocation: Distributes bankroll proportional to Sharpe ratios
    - Daily limits: Enforces bet count, stop-loss, and take-profit

    Usage:
        manager = BankrollManager(total_bankroll=1000)
        manager.load_market_performance(strategies_yaml)

        can_bet, reason = manager.can_bet_today("away_win")
        if can_bet:
            stake = manager.calculate_stake("away_win", prob=0.65, odds=2.5, edge=0.08)
    """

    def __init__(
        self,
        total_bankroll: float = 1000.0,
        config: Optional[RiskConfig] = None,
        strategies_path: Optional[Path] = None,
    ):
        """
        Initialize BankrollManager.

        Args:
            total_bankroll: Total betting bankroll in currency units
            config: Risk configuration (loads from YAML if not provided)
            strategies_path: Path to strategies.yaml for config loading
        """
        self.total_bankroll = total_bankroll
        self.config = config or RiskConfig()
        self.strategies_path = strategies_path or Path("config/strategies.yaml")

        # Market performance from walk-forward validation
        self.market_performance: Dict[str, MarketPerformance] = {}

        # Market-level bankroll allocation
        self.market_allocation: Dict[str, float] = {}

        # Daily state tracking
        self.daily_state = DailyState(date=date.today())

        # Load config from YAML if available
        if self.strategies_path.exists():
            self._load_config_from_yaml()

    def _load_config_from_yaml(self) -> None:
        """Load risk config from strategies.yaml."""
        try:
            with open(self.strategies_path) as f:
                config = yaml.safe_load(f)

            if "risk_management" in config:
                rm = config["risk_management"]
                self.config.max_daily_bets = rm.get("max_daily_bets", 10)
                self.config.max_stake_per_bet = rm.get("max_stake_per_bet", 0.05)
                self.config.stop_loss_daily = rm.get("stop_loss_daily", 0.10)
                self.config.take_profit_daily = rm.get("take_profit_daily", 0.20)

            # Load market performance from strategies
            if "strategies" in config:
                for market_name, market_config in config["strategies"].items():
                    if market_config.get("enabled", False):
                        # Extract performance metrics
                        expected_roi = market_config.get("expected_roi", 0) / 100  # Convert to decimal
                        precision = market_config.get("expected_precision", market_config.get("p_profit", 0.5))

                        # Estimate Sharpe from ROI and precision
                        # Sharpe = mean_return / std_return
                        # For betting: approx Sharpe = ROI% / sqrt(variance)
                        # Use CV (coefficient of variation) if available
                        cv = 0.3  # Default CV
                        if expected_roi > 0 and cv > 0:
                            sharpe = expected_roi / cv
                        else:
                            sharpe = 0

                        self.market_performance[market_name] = MarketPerformance(
                            market=market_name,
                            sharpe_ratio=sharpe,
                            expected_roi=expected_roi,
                            precision=precision,
                            total_bets=100,  # Placeholder
                            coefficient_of_variation=cv,
                        )

            logger.info(f"Loaded config for {len(self.market_performance)} markets")

        except Exception as e:
            logger.warning(f"Could not load strategies.yaml: {e}")

    def load_market_performance(
        self,
        performance_data: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Load market performance metrics from walk-forward validation.

        Args:
            performance_data: Dict mapping market names to performance metrics
                Example: {"away_win": {"sharpe": 1.5, "roi": 0.15, "precision": 0.65}}
        """
        for market, metrics in performance_data.items():
            self.market_performance[market] = MarketPerformance(
                market=market,
                sharpe_ratio=metrics.get("sharpe", 0),
                expected_roi=metrics.get("roi", 0),
                precision=metrics.get("precision", 0.5),
                total_bets=metrics.get("bets", 0),
                coefficient_of_variation=metrics.get("cv", 0.3),
            )

        # Recalculate allocations
        self.allocate_by_sharpe()

    def allocate_by_sharpe(self) -> Dict[str, float]:
        """
        Allocate bankroll proportional to market Sharpe ratios.

        Markets with higher Sharpe ratios receive larger allocations.
        Only profitable markets (Sharpe > 0) receive allocation.

        Returns:
            Dict mapping market names to bankroll allocations
        """
        profitable_markets = {
            name: perf for name, perf in self.market_performance.items()
            if perf.is_profitable
        }

        if not profitable_markets:
            logger.warning("No profitable markets found for allocation")
            self.market_allocation = {}
            return {}

        # Sum of positive Sharpe ratios
        total_sharpe = sum(p.sharpe_ratio for p in profitable_markets.values())

        if total_sharpe <= 0:
            # Equal allocation if no positive Sharpe
            equal_share = self.total_bankroll / len(profitable_markets)
            self.market_allocation = {name: equal_share for name in profitable_markets}
        else:
            # Proportional allocation by Sharpe
            self.market_allocation = {
                name: (perf.sharpe_ratio / total_sharpe) * self.total_bankroll
                for name, perf in profitable_markets.items()
            }

        logger.info(f"Bankroll allocation: {self.market_allocation}")
        return self.market_allocation

    def can_bet_today(self, market: str) -> Tuple[bool, str]:
        """
        Check if we can place a bet today.

        Enforces:
        - Max daily bets limit
        - Stop-loss trigger
        - Take-profit lock
        - Market profitability check

        Args:
            market: Market name (e.g., "away_win", "btts")

        Returns:
            Tuple of (can_bet: bool, reason: str)
        """
        # Reset if new day
        self.daily_state.reset_if_new_day()

        # Check daily bet limit
        if self.daily_state.bets_placed >= self.config.max_daily_bets:
            return False, f"Daily bet limit reached ({self.config.max_daily_bets})"

        # Check stop-loss
        max_loss = self.total_bankroll * self.config.stop_loss_daily
        if self.daily_state.total_pnl <= -max_loss:
            return False, f"Daily stop-loss triggered (loss: {abs(self.daily_state.total_pnl):.2f})"

        # Check take-profit
        max_profit = self.total_bankroll * self.config.take_profit_daily
        if self.daily_state.total_pnl >= max_profit:
            return False, f"Daily take-profit reached (profit: {self.daily_state.total_pnl:.2f})"

        # Check market is known and profitable
        if market in self.market_performance:
            if not self.market_performance[market].is_profitable:
                return False, f"Market {market} is not profitable"
        else:
            logger.warning(f"Unknown market: {market}, allowing bet")

        return True, "OK"

    def calculate_stake(
        self,
        market: str,
        probability: float,
        odds: float,
        edge: float,
    ) -> float:
        """
        Calculate stake using Sharpe-dampened Kelly criterion.

        Formula: stake = bankroll_market × kelly_fraction × edge × (Sharpe / dampening_factor)

        The Sharpe dampening reduces stakes for markets with high reported Sharpe,
        which may be overstated due to:
        - Small sample sizes
        - Favorable market conditions
        - Optimization bias

        Args:
            market: Market name
            probability: Model's predicted probability
            odds: Betting odds
            edge: Calculated edge (probability - implied_probability)

        Returns:
            Recommended stake in currency units
        """
        # Check minimum edge requirement
        if edge < self.config.min_edge:
            logger.debug(f"Edge {edge:.3f} below minimum {self.config.min_edge}")
            return 0.0

        # Get market-specific bankroll
        market_bankroll = self.market_allocation.get(market, self.total_bankroll * 0.1)

        # Get market Sharpe (use 1.0 as default if unknown)
        sharpe = 1.0
        if market in self.market_performance:
            sharpe = max(self.market_performance[market].sharpe_ratio, 0.1)

        # Full Kelly: f* = (p * b - 1) / (b - 1) where b = odds, p = probability
        if odds <= 1:
            return 0.0

        full_kelly = (probability * odds - 1) / (odds - 1)

        if full_kelly <= 0:
            return 0.0

        # Sharpe-dampened Kelly:
        # stake = bankroll × kelly_fraction × edge × (Sharpe / dampening_factor)
        # Higher Sharpe means more certainty → higher stake, but dampened to prevent overconfidence
        sharpe_factor = min(sharpe / self.config.sharpe_dampening_factor, 1.0)

        # Final stake calculation
        stake = market_bankroll * self.config.kelly_fraction * full_kelly * sharpe_factor

        # Cap at max stake per bet
        max_stake = self.total_bankroll * self.config.max_stake_per_bet
        stake = min(stake, max_stake)

        # Ensure positive and reasonable
        stake = max(0, stake)

        logger.debug(
            f"Stake for {market}: {stake:.2f} "
            f"(bankroll={market_bankroll:.0f}, kelly={full_kelly:.3f}, "
            f"sharpe_factor={sharpe_factor:.3f})"
        )

        return stake

    def record_bet(
        self,
        market: str,
        stake: float,
        pnl: Optional[float] = None
    ) -> None:
        """
        Record a bet for daily tracking.

        Args:
            market: Market name
            stake: Bet stake
            pnl: Profit/loss (None if bet is pending)
        """
        self.daily_state.reset_if_new_day()

        self.daily_state.bets_placed += 1
        self.daily_state.total_stake += stake

        if market not in self.daily_state.bets_by_market:
            self.daily_state.bets_by_market[market] = 0
        self.daily_state.bets_by_market[market] += 1

        if pnl is not None:
            self.daily_state.total_pnl += pnl

    def update_pnl(self, pnl: float) -> None:
        """Update daily P&L when bet settles."""
        self.daily_state.total_pnl += pnl

    def get_daily_summary(self) -> Dict[str, Any]:
        """Get summary of daily betting activity."""
        return {
            "date": str(self.daily_state.date),
            "bets_placed": self.daily_state.bets_placed,
            "max_daily_bets": self.config.max_daily_bets,
            "remaining_bets": max(0, self.config.max_daily_bets - self.daily_state.bets_placed),
            "total_stake": self.daily_state.total_stake,
            "total_pnl": self.daily_state.total_pnl,
            "pnl_percentage": (self.daily_state.total_pnl / self.total_bankroll * 100)
            if self.total_bankroll > 0 else 0,
            "stop_loss_triggered": self.daily_state.total_pnl <= -(
                self.total_bankroll * self.config.stop_loss_daily
            ),
            "take_profit_reached": self.daily_state.total_pnl >= (
                self.total_bankroll * self.config.take_profit_daily
            ),
            "bets_by_market": self.daily_state.bets_by_market.copy(),
        }

    def get_market_allocation_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of market allocations."""
        return {
            market: {
                "allocation": alloc,
                "percentage": (alloc / self.total_bankroll * 100) if self.total_bankroll > 0 else 0,
                "sharpe": self.market_performance.get(market, MarketPerformance(market, 0, 0, 0, 0)).sharpe_ratio,
                "expected_roi": self.market_performance.get(market, MarketPerformance(market, 0, 0, 0, 0)).expected_roi,
            }
            for market, alloc in self.market_allocation.items()
        }


def create_bankroll_manager(
    bankroll: float = 1000.0,
    strategies_path: Optional[str] = None,
) -> BankrollManager:
    """
    Factory function to create a configured BankrollManager.

    Args:
        bankroll: Total bankroll amount
        strategies_path: Path to strategies.yaml

    Returns:
        Configured BankrollManager instance
    """
    path = Path(strategies_path) if strategies_path else Path("config/strategies.yaml")
    manager = BankrollManager(total_bankroll=bankroll, strategies_path=path)
    manager.allocate_by_sharpe()
    return manager
