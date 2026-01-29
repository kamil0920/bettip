"""Tests for PortfolioSelector."""
from dataclasses import dataclass
from typing import Optional

import pytest

from src.recommendations.portfolio_selector import PortfolioConfig, PortfolioSelector


@dataclass
class FakeRec:
    """Minimal recommendation for testing."""
    fixture_id: int
    market: str
    our_prob: float
    odds: Optional[float] = None
    edge: Optional[float] = None
    league: str = "premier_league"


def _make_recs(n: int, **overrides) -> list:
    """Create n fake recommendations with decreasing edge."""
    recs = []
    for i in range(n):
        kwargs = {
            "fixture_id": 100 + i,
            "market": "SHOTS",
            "our_prob": 0.8 - i * 0.01,
            "odds": 2.0,
            "league": "premier_league",
        }
        kwargs.update(overrides)
        # Vary fixture_id unless overridden
        if "fixture_id" not in overrides:
            kwargs["fixture_id"] = 100 + i
        recs.append(FakeRec(**kwargs))
    return recs


class TestPortfolioSelector:
    def test_empty_input(self):
        selector = PortfolioSelector(PortfolioConfig())
        assert selector.select([]) == []

    def test_fewer_than_max(self):
        """When fewer bets than max, all are selected."""
        selector = PortfolioSelector(PortfolioConfig(max_daily_singles=10))
        recs = _make_recs(3)
        result = selector.select(recs)
        assert len(result) == 3

    def test_max_daily_singles(self):
        """Caps at max_daily_singles."""
        selector = PortfolioSelector(PortfolioConfig(max_daily_singles=5))
        recs = _make_recs(20)
        result = selector.select(recs)
        assert len(result) == 5

    def test_max_per_match(self):
        """Enforces per-match limit."""
        selector = PortfolioSelector(PortfolioConfig(max_per_match=2, max_daily_singles=10))
        recs = _make_recs(5, fixture_id=999)
        result = selector.select(recs)
        assert len(result) == 2

    def test_max_per_market(self):
        """Enforces per-market limit."""
        selector = PortfolioSelector(PortfolioConfig(max_per_market=3, max_daily_singles=10))
        recs = _make_recs(6, market="BTTS")
        result = selector.select(recs)
        assert len(result) == 3

    def test_selects_highest_edge_first(self):
        """Highest edge bets are selected first."""
        selector = PortfolioSelector(PortfolioConfig(max_daily_singles=2))
        recs = [
            FakeRec(fixture_id=1, market="SHOTS", our_prob=0.6, odds=2.0, league="la_liga"),
            FakeRec(fixture_id=2, market="BTTS", our_prob=0.9, odds=2.0, league="serie_a"),
            FakeRec(fixture_id=3, market="FOULS", our_prob=0.7, odds=2.0, league="bundesliga"),
        ]
        result = selector.select(recs)
        assert len(result) == 2
        # Highest edges: 0.9-0.5=0.4, 0.7-0.5=0.2, 0.6-0.5=0.1
        assert result[0].our_prob == 0.9
        assert result[1].our_prob == 0.7

    def test_min_leagues_soft_constraint(self):
        """Swaps lowest-edge bet to satisfy min_leagues when possible."""
        selector = PortfolioSelector(PortfolioConfig(max_daily_singles=3, min_leagues=2))
        recs = [
            FakeRec(fixture_id=1, market="SHOTS", our_prob=0.9, odds=2.0, league="epl"),
            FakeRec(fixture_id=2, market="BTTS", our_prob=0.85, odds=2.0, league="epl"),
            FakeRec(fixture_id=3, market="FOULS", our_prob=0.8, odds=2.0, league="epl"),
            FakeRec(fixture_id=4, market="SHOTS", our_prob=0.5, odds=2.0, league="la_liga"),
        ]
        result = selector.select(recs)
        leagues = {r.league for r in result}
        assert len(leagues) >= 2
        assert "la_liga" in leagues

    def test_min_leagues_relaxed_when_impossible(self):
        """min_leagues is relaxed when only one league exists."""
        selector = PortfolioSelector(PortfolioConfig(max_daily_singles=5, min_leagues=2))
        recs = _make_recs(3, league="only_league")
        result = selector.select(recs)
        assert len(result) == 3
        assert all(r.league == "only_league" for r in result)

    def test_single_bet(self):
        """Single bet skips min_leagues enforcement."""
        selector = PortfolioSelector(PortfolioConfig(min_leagues=2))
        recs = _make_recs(1)
        result = selector.select(recs)
        assert len(result) == 1

    def test_combined_constraints(self):
        """Multiple constraints work together."""
        selector = PortfolioSelector(PortfolioConfig(
            max_daily_singles=4,
            max_per_match=1,
            max_per_market=2,
        ))
        recs = [
            FakeRec(fixture_id=1, market="SHOTS", our_prob=0.9, odds=2.0),
            FakeRec(fixture_id=1, market="BTTS", our_prob=0.85, odds=2.0),  # same match
            FakeRec(fixture_id=2, market="SHOTS", our_prob=0.8, odds=2.0),
            FakeRec(fixture_id=3, market="SHOTS", our_prob=0.75, odds=2.0),  # 3rd SHOTS
            FakeRec(fixture_id=4, market="BTTS", our_prob=0.7, odds=2.0),
        ]
        result = selector.select(recs)
        # fixture 1 SHOTS (0.9), fixture 2 SHOTS (0.8), fixture 1 blocked (max_per_match=1),
        # fixture 3 SHOTS blocked (max_per_market=2), fixture 4 BTTS (0.7)
        assert len(result) <= 4
        match_ids = [r.fixture_id for r in result]
        assert match_ids.count(1) <= 1

    def test_works_with_edge_attribute(self):
        """Works with BetRecommendation-style objects that have .edge directly."""
        @dataclass
        class BetRec:
            fixture_id: str
            bet_type: str
            edge: float
            league: str

        selector = PortfolioSelector(PortfolioConfig(max_daily_singles=2))
        recs = [
            BetRec(fixture_id="f1", bet_type="btts", edge=0.1, league="epl"),
            BetRec(fixture_id="f2", bet_type="shots", edge=0.3, league="la_liga"),
            BetRec(fixture_id="f3", bet_type="fouls", edge=0.2, league="serie_a"),
        ]
        result = selector.select(recs)
        assert len(result) == 2
        assert result[0].edge == 0.3

    def test_config_from_dict(self):
        cfg = PortfolioConfig.from_dict({
            "max_daily_singles": 7,
            "max_per_match": 2,
            "max_per_market": 4,
            "min_leagues": 3,
        })
        assert cfg.max_daily_singles == 7
        assert cfg.max_per_match == 2
        assert cfg.max_per_market == 4
        assert cfg.min_leagues == 3

    def test_config_from_empty_dict(self):
        cfg = PortfolioConfig.from_dict({})
        assert cfg.max_daily_singles == 10
        assert cfg.min_leagues == 2
