"""
Portfolio Selector

Selects a diversified daily portfolio of singles from all qualifying recommendations.
Uses greedy selection with constraints to maximize edge while maintaining diversification.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PortfolioConfig:
    """Configuration for portfolio selection constraints."""
    max_daily_singles: int = 10
    max_per_match: int = 3
    max_per_market: int = 5
    min_leagues: int = 2  # Soft constraint — relaxed if not enough bets

    @classmethod
    def from_dict(cls, d: Dict) -> "PortfolioConfig":
        """Create from dictionary (e.g. YAML config)."""
        return cls(
            max_daily_singles=d.get("max_daily_singles", 10),
            max_per_match=d.get("max_per_match", 3),
            max_per_market=d.get("max_per_market", 5),
            min_leagues=d.get("min_leagues", 2),
        )


def _get_fixture_id(rec: Any) -> Any:
    """Get fixture ID from recommendation (works with both Recommendation and BetRecommendation)."""
    return rec.fixture_id


def _get_market(rec: Any) -> str:
    """Get market/bet_type from recommendation."""
    return getattr(rec, "market", None) or getattr(rec, "bet_type", "unknown")


def _get_edge(rec: Any) -> float:
    """Get edge value for sorting."""
    # BetRecommendation has .edge directly
    if hasattr(rec, "edge") and rec.edge is not None:
        return rec.edge
    # Recommendation: compute from our_prob and odds
    if hasattr(rec, "our_prob") and hasattr(rec, "odds") and rec.odds and rec.odds > 0:
        return rec.our_prob - (1.0 / rec.odds)
    if hasattr(rec, "our_prob"):
        return rec.our_prob
    return 0.0


def _get_league(rec: Any) -> str:
    """Get league from recommendation."""
    return getattr(rec, "league", "unknown")


class PortfolioSelector:
    """
    Greedy portfolio selector with diversification constraints.

    Works with both Recommendation and BetRecommendation dataclasses.

    Algorithm:
    1. Sort recommendations by edge descending
    2. Greedily accept bets that satisfy per-match and per-market limits
    3. Enforce min_leagues as a soft constraint (relax if insufficient bets)
    """

    def __init__(self, config: PortfolioConfig):
        self.config = config

    def select(self, recommendations: List[T]) -> List[T]:
        """
        Select a diversified portfolio from candidate recommendations.

        Args:
            recommendations: All qualifying recommendations for today.

        Returns:
            Filtered list respecting portfolio constraints.
        """
        if not recommendations:
            return []

        sorted_recs = sorted(recommendations, key=_get_edge, reverse=True)

        # First pass: greedy selection with hard constraints
        selected = self._greedy_select(sorted_recs)

        # Second pass: enforce min_leagues (soft — drop lowest-edge bets if needed)
        selected = self._enforce_min_leagues(selected, sorted_recs)

        leagues = {_get_league(r) for r in selected}
        markets = {_get_market(r) for r in selected}
        logger.info(
            f"Portfolio: {len(selected)}/{len(recommendations)} bets selected "
            f"({len(leagues)} leagues, {len(markets)} markets)"
        )

        return selected

    def _greedy_select(self, sorted_recs: List[T]) -> List[T]:
        """Greedy selection respecting hard constraints."""
        selected: List[T] = []
        match_counts: Dict[Any, int] = defaultdict(int)
        market_counts: Dict[str, int] = defaultdict(int)

        for rec in sorted_recs:
            if len(selected) >= self.config.max_daily_singles:
                break

            fid = _get_fixture_id(rec)
            market = _get_market(rec)

            if match_counts[fid] >= self.config.max_per_match:
                continue
            if market_counts[market] >= self.config.max_per_market:
                continue

            selected.append(rec)
            match_counts[fid] += 1
            market_counts[market] += 1

        return selected

    def _enforce_min_leagues(self, selected: List[T], all_recs: List[T]) -> List[T]:
        """
        Soft enforcement of min_leagues constraint.

        If fewer than min_leagues represented, swap the lowest-edge selected bet
        for the highest-edge bet from an underrepresented league.
        """
        if len(selected) <= 1:
            return selected

        leagues = {_get_league(r) for r in selected}
        if len(leagues) >= self.config.min_leagues:
            return selected

        selected_ids = set(id(r) for r in selected)
        missing_league_recs = [
            r for r in all_recs
            if id(r) not in selected_ids and _get_league(r) not in leagues
        ]

        if not missing_league_recs:
            return selected  # Only one league available

        # Swap lowest-edge selected for highest-edge from missing league
        worst = min(selected, key=_get_edge)
        best_new = missing_league_recs[0]  # all_recs already sorted by edge desc

        selected.remove(worst)
        selected.append(best_new)

        return selected
