"""Unit tests for the weekend betting backtest simulator."""

import numpy as np
import pandas as pd
import pytest

from experiments.run_weekend_backtest import (
    BacktestConfig,
    DayResult,
    WeekendResult,
    compute_aggregate_metrics,
    compute_edge,
    identify_weekends,
    run_backtest,
    select_daily_bets,
    simulate_day,
)


# --- Fixtures ---


def _make_config(**overrides) -> BacktestConfig:
    defaults = dict(
        daily_bankroll=500,
        max_bets_per_day=10,
        max_per_market=5,
        max_per_match=1,
        stake_per_bet=50,
        min_edge=0.02,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _make_bets(
    n: int,
    date: str = "2025-01-03",
    market: str = "cards_over_15",
    fixture_prefix: str = "fix",
    prob: float = 0.70,
    odds: float = 1.80,
    actual: int = 1,
    qualifies: bool = True,
) -> pd.DataFrame:
    """Create a DataFrame of synthetic bets."""
    return pd.DataFrame(
        {
            "date": pd.Timestamp(date),
            "fixture_id": [f"{fixture_prefix}_{i}" for i in range(n)],
            "league": "premier_league",
            "prob": prob,
            "odds": odds,
            "actual": actual,
            "market": market,
            "threshold": 0.6,
            "model": "catboost",
            "qualifies": qualifies,
        }
    )


# --- compute_edge ---


class TestComputeEdge:
    def test_basic_edge_calculation(self):
        df = pd.DataFrame({"prob": [0.70], "odds": [2.0]})
        result = compute_edge(df)
        assert "edge" in result.columns
        assert "implied_prob" in result.columns
        # edge = 0.70 - 1/2.0 = 0.70 - 0.50 = 0.20
        assert abs(result["edge"].iloc[0] - 0.20) < 1e-6

    def test_no_edge(self):
        df = pd.DataFrame({"prob": [0.50], "odds": [2.0]})
        result = compute_edge(df)
        assert abs(result["edge"].iloc[0]) < 1e-6

    def test_negative_edge(self):
        df = pd.DataFrame({"prob": [0.40], "odds": [2.0]})
        result = compute_edge(df)
        assert result["edge"].iloc[0] < 0

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"prob": [0.70], "odds": [2.0]})
        original_cols = list(df.columns)
        compute_edge(df)
        assert list(df.columns) == original_cols

    def test_low_odds_clipping(self):
        """Odds of 1.0 would cause division by zero — should be clipped."""
        df = pd.DataFrame({"prob": [0.90], "odds": [1.0]})
        result = compute_edge(df)
        assert np.isfinite(result["edge"].iloc[0])


# --- identify_weekends ---


class TestIdentifyWeekends:
    def test_single_weekend(self):
        dates = pd.Series(pd.to_datetime(["2025-01-03", "2025-01-04", "2025-01-05"]))
        weekends = identify_weekends(dates)
        assert len(weekends) == 1
        assert len(weekends[0]) == 3

    def test_two_weekends(self):
        dates = pd.Series(
            pd.to_datetime(
                [
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",  # Weekend 1
                    "2025-01-10",
                    "2025-01-11",
                    "2025-01-12",  # Weekend 2
                ]
            )
        )
        weekends = identify_weekends(dates)
        assert len(weekends) == 2
        assert len(weekends[0]) == 3
        assert len(weekends[1]) == 3

    def test_partial_weekend_saturday_only(self):
        dates = pd.Series(pd.to_datetime(["2025-01-04"]))  # Saturday only
        weekends = identify_weekends(dates)
        assert len(weekends) == 1
        assert len(weekends[0]) == 1

    def test_no_weekend_dates(self):
        # Monday-Thursday only
        dates = pd.Series(pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08"]))
        weekends = identify_weekends(dates)
        assert len(weekends) == 0

    def test_empty_series(self):
        dates = pd.Series(dtype="datetime64[ns]")
        weekends = identify_weekends(dates)
        assert len(weekends) == 0

    def test_duplicate_dates(self):
        dates = pd.Series(
            pd.to_datetime(["2025-01-03", "2025-01-03", "2025-01-04", "2025-01-04"])
        )
        weekends = identify_weekends(dates)
        assert len(weekends) == 1
        assert len(weekends[0]) == 2  # Fri + Sat, deduplicated

    def test_friday_sunday_gap(self):
        """Friday and Sunday without Saturday should still be one weekend."""
        dates = pd.Series(pd.to_datetime(["2025-01-03", "2025-01-05"]))
        weekends = identify_weekends(dates)
        assert len(weekends) == 1
        assert len(weekends[0]) == 2


# --- select_daily_bets ---


class TestSelectDailyBets:
    def test_max_bets_per_day(self):
        config = _make_config(max_bets_per_day=3)
        pool = _make_bets(10, prob=0.75, odds=1.80)
        pool = compute_edge(pool)
        pool = pool.sort_values("edge", ascending=False)
        selected = select_daily_bets(pool, config)
        assert len(selected) == 3

    def test_max_per_market(self):
        config = _make_config(max_bets_per_day=10, max_per_market=2)
        pool = _make_bets(10, market="cards_over_15", prob=0.75, odds=1.80)
        pool = compute_edge(pool)
        pool = pool.sort_values("edge", ascending=False)
        selected = select_daily_bets(pool, config)
        assert len(selected) == 2

    def test_max_per_match(self):
        config = _make_config(max_bets_per_day=10, max_per_match=1)
        # All same fixture_id
        pool = pd.DataFrame(
            {
                "fixture_id": ["match_1"] * 5,
                "market": [f"market_{i}" for i in range(5)],
                "prob": 0.75,
                "odds": 1.80,
                "edge": 0.20,
            }
        )
        selected = select_daily_bets(pool, config)
        assert len(selected) == 1

    def test_min_edge_filter(self):
        config = _make_config(min_edge=0.10)
        pool = pd.DataFrame(
            {
                "fixture_id": [f"fix_{i}" for i in range(5)],
                "market": "cards_over_15",
                "prob": [0.55, 0.60, 0.65, 0.70, 0.75],
                "odds": 2.0,
                "edge": [0.05, 0.10, 0.15, 0.20, 0.25],
            }
        )
        pool = pool.sort_values("edge", ascending=False)
        selected = select_daily_bets(pool, config)
        # Only edges >= 0.10 qualify: 0.10, 0.15, 0.20, 0.25
        assert len(selected) == 4

    def test_empty_pool(self):
        config = _make_config()
        pool = pd.DataFrame(
            columns=["fixture_id", "market", "prob", "odds", "edge"]
        )
        selected = select_daily_bets(pool, config)
        assert len(selected) == 0

    def test_mixed_markets_respects_limit(self):
        config = _make_config(max_bets_per_day=10, max_per_market=2)
        # 5 bets from 3 markets
        rows = []
        for i, m in enumerate(["cards_over_15"] * 4 + ["corners_over_85"] * 4 + ["shots_under_285"] * 4):
            rows.append(
                {
                    "fixture_id": f"fix_{i}",
                    "market": m,
                    "prob": 0.75,
                    "odds": 1.80,
                    "edge": 0.20 - i * 0.001,
                }
            )
        pool = pd.DataFrame(rows)
        pool = pool.sort_values("edge", ascending=False)
        selected = select_daily_bets(pool, config)
        # max 2 per market × 3 markets = 6, but pool has 12 so limited to 6
        assert len(selected) == 6
        market_counts = selected["market"].value_counts()
        assert market_counts.max() <= 2

    def test_selects_highest_edge_first(self):
        config = _make_config(max_bets_per_day=2)
        pool = pd.DataFrame(
            {
                "fixture_id": [f"fix_{i}" for i in range(5)],
                "market": "cards_over_15",
                "prob": [0.60, 0.70, 0.80, 0.65, 0.75],
                "odds": 2.0,
                "edge": [0.10, 0.20, 0.30, 0.15, 0.25],
            }
        )
        pool = pool.sort_values("edge", ascending=False)
        selected = select_daily_bets(pool, config)
        assert list(selected["edge"]) == [0.30, 0.25]


# --- simulate_day ---


class TestSimulateDay:
    def test_all_wins(self):
        config = _make_config(stake_per_bet=50)
        bets = _make_bets(3, actual=1, odds=2.0)
        bets = compute_edge(bets)
        result = simulate_day(bets, config, pd.Timestamp("2025-01-03"))
        assert result.bets_placed == 3
        assert result.wins == 3
        assert result.pnl == 3 * 50 * (2.0 - 1)  # 150
        assert result.staked == 150

    def test_all_losses(self):
        config = _make_config(stake_per_bet=50)
        bets = _make_bets(3, actual=0, odds=2.0)
        bets = compute_edge(bets)
        result = simulate_day(bets, config, pd.Timestamp("2025-01-03"))
        assert result.wins == 0
        assert result.pnl == -150

    def test_mixed_results(self):
        config = _make_config(stake_per_bet=50)
        bets = pd.DataFrame(
            {
                "fixture_id": ["a", "b", "c"],
                "market": "cards_over_15",
                "actual": [1, 0, 1],
                "odds": [2.0, 1.80, 2.50],
                "edge": [0.10, 0.10, 0.10],
                "league": "premier_league",
            }
        )
        result = simulate_day(bets, config, pd.Timestamp("2025-01-04"))
        assert result.bets_placed == 3
        assert result.wins == 2
        # Win: 50*(2.0-1) + Loss: -50 + Win: 50*(2.5-1) = 50 - 50 + 75 = 75
        assert abs(result.pnl - 75.0) < 1e-6
        assert result.day_of_week == "Saturday"

    def test_empty_bets(self):
        config = _make_config()
        bets = pd.DataFrame(
            columns=["fixture_id", "market", "actual", "odds", "edge", "league"]
        )
        result = simulate_day(bets, config, pd.Timestamp("2025-01-03"))
        assert result.bets_placed == 0
        assert result.pnl == 0.0


# --- run_backtest ---


class TestRunBacktest:
    def _make_weekend_df(self) -> pd.DataFrame:
        """Create a minimal 1-weekend dataset."""
        rows = []
        for date in ["2025-01-03", "2025-01-04", "2025-01-05"]:
            for i in range(5):
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "fixture_id": f"fix_{date}_{i}",
                        "league": "premier_league",
                        "prob": 0.70,
                        "odds": 1.80,
                        "actual": 1 if i < 3 else 0,
                        "market": "cards_over_15",
                        "threshold": 0.6,
                        "model": "catboost",
                        "qualifies": True,
                    }
                )
        return pd.DataFrame(rows)

    def test_basic_backtest(self):
        df = self._make_weekend_df()
        config = _make_config(max_bets_per_day=5)
        results = run_backtest(df, config)
        assert len(results) == 1
        assert len(results[0].days) == 3
        assert results[0].total_bets > 0

    def test_non_qualifying_excluded(self):
        df = self._make_weekend_df()
        df["qualifies"] = False
        config = _make_config()
        results = run_backtest(df, config)
        # No qualifying bets means no weekends (no weekend dates)
        assert len(results) == 0

    def test_weekday_bets_ignored(self):
        """Bets on weekdays should not appear in weekend results."""
        rows = []
        # Monday bets only
        for i in range(5):
            rows.append(
                {
                    "date": pd.Timestamp("2025-01-06"),  # Monday
                    "fixture_id": f"fix_{i}",
                    "league": "premier_league",
                    "prob": 0.70,
                    "odds": 1.80,
                    "actual": 1,
                    "market": "cards_over_15",
                    "threshold": 0.6,
                    "model": "catboost",
                    "qualifies": True,
                }
            )
        df = pd.DataFrame(rows)
        config = _make_config()
        results = run_backtest(df, config)
        assert len(results) == 0


# --- compute_aggregate_metrics ---


class TestAggregateMetrics:
    def _make_weekend_results(self) -> list:
        """Create 3 synthetic weekend results."""
        weekends = []
        for i, (pnl, bets, wins) in enumerate(
            [(100, 20, 14), (-200, 25, 8), (50, 15, 10)]
        ):
            staked = bets * 50
            days = [
                DayResult(
                    date=f"2025-01-0{3 + i * 7}",
                    day_of_week="Friday",
                    bets_placed=bets // 3,
                    wins=wins // 3,
                    pnl=pnl / 3,
                    staked=staked / 3,
                ),
                DayResult(
                    date=f"2025-01-0{4 + i * 7}",
                    day_of_week="Saturday",
                    bets_placed=bets // 3,
                    wins=wins // 3,
                    pnl=pnl / 3,
                    staked=staked / 3,
                ),
                DayResult(
                    date=f"2025-01-0{5 + i * 7}",
                    day_of_week="Sunday",
                    bets_placed=bets - 2 * (bets // 3),
                    wins=wins - 2 * (wins // 3),
                    pnl=pnl - 2 * (pnl / 3),
                    staked=staked - 2 * (staked / 3),
                ),
            ]
            weekends.append(
                WeekendResult(
                    weekend_label=f"Weekend {i + 1}",
                    days=days,
                    total_pnl=pnl,
                    total_staked=staked,
                    roi=pnl / staked * 100,
                    total_bets=bets,
                    total_wins=wins,
                )
            )
        return weekends

    def test_basic_metrics(self):
        results = self._make_weekend_results()
        metrics = compute_aggregate_metrics(results)
        assert metrics["n_weekends"] == 3
        assert metrics["total_bets"] == 60
        assert metrics["total_wins"] == 32
        assert abs(metrics["total_pnl"] - (-50.0)) < 1e-6

    def test_positive_weekend_count(self):
        results = self._make_weekend_results()
        metrics = compute_aggregate_metrics(results)
        # Weekends 1 (+100) and 3 (+50) are positive
        assert metrics["positive_weekends"] == 2

    def test_worst_best_weekend(self):
        results = self._make_weekend_results()
        metrics = compute_aggregate_metrics(results)
        assert metrics["worst_weekend_pnl"] == -200
        assert metrics["best_weekend_pnl"] == 100

    def test_max_drawdown(self):
        results = self._make_weekend_results()
        metrics = compute_aggregate_metrics(results)
        # Cumulative: 100, -100, -50
        # Peak: 100, drawdown from peak to -100 = 200
        assert metrics["max_drawdown"] == 200.0

    def test_empty_results(self):
        metrics = compute_aggregate_metrics([])
        assert "error" in metrics

    def test_day_of_week_breakdown(self):
        results = self._make_weekend_results()
        metrics = compute_aggregate_metrics(results)
        assert "Friday" in metrics["day_of_week_pnl"]
        assert "Saturday" in metrics["day_of_week_pnl"]
        assert "Sunday" in metrics["day_of_week_pnl"]

    def test_weekend_sharpe(self):
        results = self._make_weekend_results()
        metrics = compute_aggregate_metrics(results)
        # With 3 weekends, Sharpe should be finite
        assert np.isfinite(metrics["weekend_sharpe"])
