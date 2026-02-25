"""
Weekend Betting Simulator — Realistic Model Validation

Replays the holdout period weekend-by-weekend under real-world constraints:
- 3 betting days per weekend (Fri/Sat/Sun)
- 10 single bets per day, max 1 per match, max 5 per market
- Flat staking (default 50 PLN/bet, 500 PLN/day)
- Cross-market portfolio selection by edge ranking

Usage:
    python experiments/run_weekend_backtest.py \
        --holdout-dir experiments/outputs/sniper_optimization/ \
        --daily-bankroll 500 --max-bets-per-day 10 --max-per-market 5 \
        --stake-per-bet 50 --min-edge 0.02
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.metrics import sharpe_ratio

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("experiments/outputs/weekend_backtest")


@dataclass
class BacktestConfig:
    """Configuration for weekend backtest simulation."""

    daily_bankroll: float = 500.0
    max_bets_per_day: int = 10
    max_per_market: int = 5
    max_per_match: int = 1
    stake_per_bet: float = 50.0
    min_edge: float = 0.02


@dataclass
class DayResult:
    """Results for a single betting day."""

    date: str
    day_of_week: str
    bets_placed: int
    wins: int
    pnl: float
    staked: float
    markets_used: List[str] = field(default_factory=list)
    leagues_used: List[str] = field(default_factory=list)
    avg_edge: float = 0.0
    avg_odds: float = 0.0


@dataclass
class WeekendResult:
    """Results for a full weekend (Fri+Sat+Sun)."""

    weekend_label: str
    days: List[DayResult]
    total_pnl: float = 0.0
    total_staked: float = 0.0
    roi: float = 0.0
    total_bets: int = 0
    total_wins: int = 0


def load_holdout_predictions(holdout_dir: str) -> pd.DataFrame:
    """Load and combine holdout prediction CSVs from all markets.

    Args:
        holdout_dir: Directory containing holdout_preds_*.csv files.

    Returns:
        Combined DataFrame with columns:
        date, fixture_id, league, prob, odds, actual, market, threshold, qualifies
    """
    holdout_path = Path(holdout_dir)
    csv_files = sorted(holdout_path.glob("holdout_preds_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No holdout_preds_*.csv files found in {holdout_path}"
        )

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        logger.info(f"  Loaded {f.name}: {len(df)} rows, {df['qualifies'].sum()} qualifying")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    logger.info(
        f"Combined: {len(combined)} holdout predictions from {len(csv_files)} markets, "
        f"{combined['qualifies'].sum()} qualifying bets"
    )
    return combined


def compute_edge(df: pd.DataFrame) -> pd.DataFrame:
    """Compute edge per bet: prob - implied_probability."""
    df = df.copy()
    df["implied_prob"] = 1.0 / df["odds"].clip(lower=1.01)
    df["edge"] = df["prob"] - df["implied_prob"]
    return df


def identify_weekends(dates: pd.Series) -> List[List[pd.Timestamp]]:
    """Group dates into weekend blocks (Fri=4, Sat=5, Sun=6).

    Returns:
        List of weekends, each a list of dates present in that weekend.
    """
    weekend_dates = dates[dates.dt.dayofweek.isin([4, 5, 6])].drop_duplicates().sort_values()

    if weekend_dates.empty:
        return []

    weekends: List[List[pd.Timestamp]] = []
    current_weekend: List[pd.Timestamp] = []

    for d in weekend_dates:
        if not current_weekend:
            current_weekend.append(d)
        else:
            # Same weekend if within 2 days of last date and still Fri/Sat/Sun
            gap = (d - current_weekend[-1]).days
            if gap <= 2 and d.dayofweek in [4, 5, 6]:
                current_weekend.append(d)
            else:
                weekends.append(current_weekend)
                current_weekend = [d]

    if current_weekend:
        weekends.append(current_weekend)

    return weekends


def select_daily_bets(
    day_pool: pd.DataFrame, config: BacktestConfig
) -> pd.DataFrame:
    """Greedy selection of bets for a single day.

    Args:
        day_pool: All qualifying bets for this date, sorted by edge desc.
        config: Backtest configuration with constraints.

    Returns:
        Selected bets DataFrame.
    """
    selected_idx = []
    match_counts: Dict = {}
    market_counts: Dict[str, int] = {}

    for idx, row in day_pool.iterrows():
        if len(selected_idx) >= config.max_bets_per_day:
            break

        fid = row["fixture_id"]
        market = row["market"]
        edge = row["edge"]

        if edge < config.min_edge:
            continue
        if match_counts.get(fid, 0) >= config.max_per_match:
            continue
        if market_counts.get(market, 0) >= config.max_per_market:
            continue

        selected_idx.append(idx)
        match_counts[fid] = match_counts.get(fid, 0) + 1
        market_counts[market] = market_counts.get(market, 0) + 1

    return day_pool.loc[selected_idx]


def simulate_day(
    day_bets: pd.DataFrame, config: BacktestConfig, date: pd.Timestamp
) -> DayResult:
    """Simulate betting for a single day.

    Args:
        day_bets: Selected bets for this day.
        config: Backtest configuration.
        date: The date being simulated.

    Returns:
        DayResult with P&L and metadata.
    """
    day_name = date.strftime("%A")
    n_bets = len(day_bets)

    if n_bets == 0:
        return DayResult(
            date=str(date.date()),
            day_of_week=day_name,
            bets_placed=0,
            wins=0,
            pnl=0.0,
            staked=0.0,
        )

    wins = int(day_bets["actual"].sum())
    returns = np.where(
        day_bets["actual"] == 1,
        config.stake_per_bet * (day_bets["odds"] - 1),
        -config.stake_per_bet,
    )
    pnl = float(returns.sum())
    staked = n_bets * config.stake_per_bet

    return DayResult(
        date=str(date.date()),
        day_of_week=day_name,
        bets_placed=n_bets,
        wins=wins,
        pnl=pnl,
        staked=staked,
        markets_used=sorted(day_bets["market"].unique().tolist()),
        leagues_used=sorted(day_bets["league"].unique().tolist()),
        avg_edge=float(day_bets["edge"].mean()),
        avg_odds=float(day_bets["odds"].mean()),
    )


def run_backtest(
    df: pd.DataFrame, config: BacktestConfig
) -> List[WeekendResult]:
    """Run the full weekend backtest simulation.

    Args:
        df: Combined holdout predictions (all markets).
        config: Backtest configuration.

    Returns:
        List of WeekendResult objects.
    """
    # Filter to qualifying bets only
    qualifying = df[df["qualifies"]].copy()
    qualifying = compute_edge(qualifying)

    logger.info(
        f"Qualifying bets: {len(qualifying)} across "
        f"{qualifying['date'].dt.date.nunique()} unique dates"
    )

    # Identify weekends
    all_dates = qualifying["date"]
    weekends = identify_weekends(all_dates)
    logger.info(f"Identified {len(weekends)} weekends in holdout period")

    results: List[WeekendResult] = []

    for weekend_dates in weekends:
        weekend_label = f"{weekend_dates[0].date()} to {weekend_dates[-1].date()}"
        day_results: List[DayResult] = []

        for date in weekend_dates:
            # Pool all qualifying bets for this date
            day_pool = qualifying[qualifying["date"] == date].copy()
            day_pool = day_pool.sort_values("edge", ascending=False)

            # Greedy selection with constraints
            selected = select_daily_bets(day_pool, config)

            # Simulate
            day_result = simulate_day(selected, config, date)
            day_results.append(day_result)

        total_pnl = sum(d.pnl for d in day_results)
        total_staked = sum(d.staked for d in day_results)
        total_bets = sum(d.bets_placed for d in day_results)
        total_wins = sum(d.wins for d in day_results)
        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0

        results.append(
            WeekendResult(
                weekend_label=weekend_label,
                days=day_results,
                total_pnl=total_pnl,
                total_staked=total_staked,
                roi=roi,
                total_bets=total_bets,
                total_wins=total_wins,
            )
        )

    return results


def compute_aggregate_metrics(
    results: List[WeekendResult],
) -> Dict:
    """Compute aggregate metrics across all weekends.

    Args:
        results: List of WeekendResult objects.

    Returns:
        Dictionary of aggregate metrics.
    """
    if not results:
        return {"error": "No weekends simulated"}

    weekend_pnls = np.array([w.total_pnl for w in results])
    weekend_rois = np.array([w.roi for w in results])
    weekend_bets = np.array([w.total_bets for w in results])
    weekend_wins = np.array([w.total_wins for w in results])

    # Per-bet returns for Sharpe calculation
    all_returns = []
    for w in results:
        for d in w.days:
            # We don't have per-bet returns stored, reconstruct from aggregates
            if d.bets_placed > 0:
                avg_win_return = (d.pnl + d.bets_placed * (d.staked / d.bets_placed)) / max(d.wins, 1) - (d.staked / d.bets_placed) if d.wins > 0 else 0
                # Simpler: use weekend-level returns
                pass

    # Weekend-level Sharpe (using weekend ROI as returns)
    weekend_returns_pct = weekend_rois / 100.0
    if len(weekend_returns_pct) >= 2:
        std = np.std(weekend_returns_pct, ddof=1)
        weekend_sharpe = float(np.mean(weekend_returns_pct) / std) if std > 0 else 0.0
    else:
        weekend_sharpe = 0.0

    # Cumulative equity curve
    cumulative_pnl = np.cumsum(weekend_pnls)
    max_drawdown = 0.0
    peak = 0.0
    for val in cumulative_pnl:
        peak = max(peak, val)
        drawdown = peak - val
        max_drawdown = max(max_drawdown, drawdown)

    # Per-market contribution
    market_pnl: Dict[str, float] = {}
    market_bets: Dict[str, int] = {}
    for w in results:
        for d in w.days:
            for m in d.markets_used:
                market_pnl[m] = market_pnl.get(m, 0)
                market_bets[m] = market_bets.get(m, 0)

    # Per-day-of-week breakdown
    dow_pnl: Dict[str, float] = {}
    dow_bets: Dict[str, int] = {}
    for w in results:
        for d in w.days:
            dow_pnl[d.day_of_week] = dow_pnl.get(d.day_of_week, 0) + d.pnl
            dow_bets[d.day_of_week] = dow_bets.get(d.day_of_week, 0) + d.bets_placed

    # Winning weekends
    positive_weekends = int((weekend_pnls > 0).sum())

    return {
        "n_weekends": len(results),
        "total_bets": int(weekend_bets.sum()),
        "total_wins": int(weekend_wins.sum()),
        "win_rate": float(weekend_wins.sum() / weekend_bets.sum() * 100) if weekend_bets.sum() > 0 else 0.0,
        "total_pnl": float(weekend_pnls.sum()),
        "total_staked": float(sum(w.total_staked for w in results)),
        "overall_roi": float(weekend_pnls.sum() / sum(w.total_staked for w in results) * 100) if sum(w.total_staked for w in results) > 0 else 0.0,
        "mean_weekend_pnl": float(weekend_pnls.mean()),
        "median_weekend_pnl": float(np.median(weekend_pnls)),
        "std_weekend_pnl": float(weekend_pnls.std(ddof=1)) if len(weekend_pnls) > 1 else 0.0,
        "mean_weekend_roi": float(weekend_rois.mean()),
        "median_weekend_roi": float(np.median(weekend_rois)),
        "best_weekend_pnl": float(weekend_pnls.max()),
        "worst_weekend_pnl": float(weekend_pnls.min()),
        "best_weekend_roi": float(weekend_rois.max()),
        "worst_weekend_roi": float(weekend_rois.min()),
        "positive_weekends": positive_weekends,
        "positive_weekend_pct": float(positive_weekends / len(results) * 100),
        "weekend_sharpe": weekend_sharpe,
        "max_drawdown": float(max_drawdown),
        "mean_bets_per_weekend": float(weekend_bets.mean()),
        "day_of_week_pnl": dow_pnl,
        "day_of_week_bets": dow_bets,
    }


def compute_per_market_metrics(
    df: pd.DataFrame, results: List[WeekendResult], config: BacktestConfig
) -> Dict[str, Dict]:
    """Compute per-market contribution metrics from the raw holdout data.

    Re-runs selection to track which market each bet came from.
    """
    qualifying = df[df["qualifies"]].copy()
    qualifying = compute_edge(qualifying)

    all_dates = qualifying["date"]
    weekends = identify_weekends(all_dates)

    market_stats: Dict[str, Dict] = {}

    for weekend_dates in weekends:
        for date in weekend_dates:
            day_pool = qualifying[qualifying["date"] == date].copy()
            day_pool = day_pool.sort_values("edge", ascending=False)
            selected = select_daily_bets(day_pool, config)

            for _, row in selected.iterrows():
                m = row["market"]
                if m not in market_stats:
                    market_stats[m] = {"bets": 0, "wins": 0, "pnl": 0.0, "total_edge": 0.0}

                market_stats[m]["bets"] += 1
                market_stats[m]["total_edge"] += row["edge"]
                if row["actual"] == 1:
                    market_stats[m]["wins"] += 1
                    market_stats[m]["pnl"] += config.stake_per_bet * (row["odds"] - 1)
                else:
                    market_stats[m]["pnl"] -= config.stake_per_bet

    # Compute derived metrics
    for m, stats in market_stats.items():
        n = stats["bets"]
        stats["win_rate"] = stats["wins"] / n * 100 if n > 0 else 0.0
        stats["roi"] = stats["pnl"] / (n * config.stake_per_bet) * 100 if n > 0 else 0.0
        stats["avg_edge"] = stats["total_edge"] / n if n > 0 else 0.0
        del stats["total_edge"]

    return market_stats


def print_report(
    results: List[WeekendResult],
    metrics: Dict,
    market_metrics: Dict[str, Dict],
    config: BacktestConfig,
) -> None:
    """Print formatted backtest report to console."""
    print("\n" + "=" * 80)
    print("WEEKEND BETTING BACKTEST RESULTS")
    print("=" * 80)
    print(
        f"\nConfig: {config.max_bets_per_day} bets/day, "
        f"{config.stake_per_bet} PLN/bet, "
        f"max {config.max_per_market}/market, "
        f"max {config.max_per_match}/match, "
        f"min edge {config.min_edge:.1%}"
    )

    # Overall summary
    print(f"\n--- Overall ({metrics['n_weekends']} weekends) ---")
    print(f"Total bets:     {metrics['total_bets']}")
    print(f"Total staked:   {metrics['total_staked']:.0f} PLN")
    print(f"Total P&L:      {metrics['total_pnl']:+.2f} PLN")
    print(f"Overall ROI:    {metrics['overall_roi']:+.1f}%")
    print(f"Win rate:       {metrics['win_rate']:.1f}%")
    print(f"Weekend Sharpe: {metrics['weekend_sharpe']:.3f}")
    print(f"Max drawdown:   {metrics['max_drawdown']:.2f} PLN")

    # Weekend distribution
    print(f"\n--- Weekend Distribution ---")
    print(f"Mean P&L:       {metrics['mean_weekend_pnl']:+.2f} PLN")
    print(f"Median P&L:     {metrics['median_weekend_pnl']:+.2f} PLN")
    print(f"Std P&L:        {metrics['std_weekend_pnl']:.2f} PLN")
    print(f"Best weekend:   {metrics['best_weekend_pnl']:+.2f} PLN ({metrics['best_weekend_roi']:+.1f}%)")
    print(f"Worst weekend:  {metrics['worst_weekend_pnl']:+.2f} PLN ({metrics['worst_weekend_roi']:+.1f}%)")
    print(f"Positive:       {metrics['positive_weekends']}/{metrics['n_weekends']} ({metrics['positive_weekend_pct']:.0f}%)")
    print(f"Mean bets/wknd: {metrics['mean_bets_per_weekend']:.1f}")

    # Day of week
    print(f"\n--- Day of Week ---")
    for dow in ["Friday", "Saturday", "Sunday"]:
        pnl = metrics["day_of_week_pnl"].get(dow, 0)
        bets = metrics["day_of_week_bets"].get(dow, 0)
        roi = pnl / (bets * config.stake_per_bet) * 100 if bets > 0 else 0
        print(f"  {dow:10s}: {bets:4d} bets, {pnl:+8.2f} PLN ({roi:+.1f}% ROI)")

    # Per-market
    print(f"\n--- Per-Market Contribution ---")
    sorted_markets = sorted(market_metrics.items(), key=lambda x: x[1]["pnl"], reverse=True)
    print(f"  {'Market':<25s} {'Bets':>5s} {'Wins':>5s} {'WR%':>6s} {'P&L':>10s} {'ROI%':>7s} {'AvgEdge':>8s}")
    for m, s in sorted_markets:
        print(
            f"  {m:<25s} {s['bets']:5d} {s['wins']:5d} {s['win_rate']:5.1f}% "
            f"{s['pnl']:+10.2f} {s['roi']:+6.1f}% {s['avg_edge']:7.3f}"
        )

    # Per-weekend detail
    print(f"\n--- Weekend Details ---")
    print(f"  {'Weekend':<30s} {'Bets':>5s} {'Wins':>5s} {'P&L':>10s} {'ROI%':>7s} {'CumP&L':>10s}")
    cum_pnl = 0.0
    for w in results:
        cum_pnl += w.total_pnl
        print(
            f"  {w.weekend_label:<30s} {w.total_bets:5d} {w.total_wins:5d} "
            f"{w.total_pnl:+10.2f} {w.roi:+6.1f}% {cum_pnl:+10.2f}"
        )

    print("=" * 80)


def save_results(
    results: List[WeekendResult],
    metrics: Dict,
    market_metrics: Dict[str, Dict],
    config: BacktestConfig,
) -> None:
    """Save backtest results to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full metrics JSON
    output = {
        "config": {
            "daily_bankroll": config.daily_bankroll,
            "max_bets_per_day": config.max_bets_per_day,
            "max_per_market": config.max_per_market,
            "max_per_match": config.max_per_match,
            "stake_per_bet": config.stake_per_bet,
            "min_edge": config.min_edge,
        },
        "aggregate_metrics": metrics,
        "per_market": market_metrics,
        "weekends": [
            {
                "label": w.weekend_label,
                "pnl": w.total_pnl,
                "roi": w.roi,
                "bets": w.total_bets,
                "wins": w.total_wins,
                "staked": w.total_staked,
                "days": [
                    {
                        "date": d.date,
                        "day_of_week": d.day_of_week,
                        "bets": d.bets_placed,
                        "wins": d.wins,
                        "pnl": d.pnl,
                        "staked": d.staked,
                        "markets": d.markets_used,
                        "leagues": d.leagues_used,
                        "avg_edge": d.avg_edge,
                        "avg_odds": d.avg_odds,
                    }
                    for d in w.days
                ],
            }
            for w in results
        ],
    }

    results_path = OUTPUT_DIR / "weekend_backtest_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved results: {results_path}")

    # Save equity curve CSV
    equity_data = []
    cum_pnl = 0.0
    for w in results:
        cum_pnl += w.total_pnl
        equity_data.append(
            {
                "weekend": w.weekend_label,
                "pnl": w.total_pnl,
                "roi": w.roi,
                "cumulative_pnl": cum_pnl,
                "bets": w.total_bets,
                "wins": w.total_wins,
            }
        )

    equity_df = pd.DataFrame(equity_data)
    equity_path = OUTPUT_DIR / "weekend_equity_curve.csv"
    equity_df.to_csv(equity_path, index=False)
    logger.info(f"Saved equity curve: {equity_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weekend Betting Backtest — Realistic Model Validation"
    )
    parser.add_argument(
        "--holdout-dir",
        type=str,
        default="experiments/outputs/sniper_optimization",
        help="Directory containing holdout_preds_*.csv files",
    )
    parser.add_argument("--daily-bankroll", type=float, default=500.0)
    parser.add_argument("--max-bets-per-day", type=int, default=10)
    parser.add_argument("--max-per-market", type=int, default=5)
    parser.add_argument("--max-per-match", type=int, default=1)
    parser.add_argument("--stake-per-bet", type=float, default=50.0)
    parser.add_argument("--min-edge", type=float, default=0.02)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = BacktestConfig(
        daily_bankroll=args.daily_bankroll,
        max_bets_per_day=args.max_bets_per_day,
        max_per_market=args.max_per_market,
        max_per_match=args.max_per_match,
        stake_per_bet=args.stake_per_bet,
        min_edge=args.min_edge,
    )

    logger.info("Loading holdout predictions...")
    df = load_holdout_predictions(args.holdout_dir)

    logger.info("Running weekend backtest simulation...")
    results = run_backtest(df, config)

    if not results:
        logger.warning("No weekends found in holdout data. Check date range.")
        return

    logger.info("Computing aggregate metrics...")
    metrics = compute_aggregate_metrics(results)
    market_metrics = compute_per_market_metrics(df, results, config)

    print_report(results, metrics, market_metrics, config)
    save_results(results, metrics, market_metrics, config)

    logger.info("Done.")


if __name__ == "__main__":
    main()
