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
from fuzzywuzzy import fuzz

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.metrics import sharpe_ratio
from src.odds.live_odds_client import parse_market_name

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("experiments/outputs/weekend_backtest")

# Mapping: stat → (total_col, home_ema_col, away_ema_col, league_avg_col)
_STAT_CONFIG = {
    "corners": ("total_corners", "home_corners_won_ema", "away_corners_won_ema", "league_avg_total_corners"),
    "cards": ("total_cards", "home_cards_ema", "away_cards_ema", "league_avg_total_cards"),
    "shots": ("total_shots", "home_shots_total_ema", "away_shots_total_ema", "league_avg_total_shots"),
    "fouls": ("total_fouls", "home_fouls_match_ema", "away_fouls_match_ema", "league_avg_total_fouls"),
}

# Handicap markets use diff columns
_HC_STAT_CONFIG = {
    "cornershc": ("home_corners", "away_corners", "home_corners_won_ema", "away_corners_conceded_ema"),
    "cardshc": ("home_cards", "away_cards", "home_cards_ema", "away_cards_ema"),
}

# Median real odds from historical collection (Aug-Oct 2024, Big 5 leagues).
# HC odds cluster tightly around these values regardless of match — bookmakers
# adjust the LINE to balance action, not the odds.
# Format: (stat, line) → {"over": median_over_avg, "under": median_under_avg}
_HC_REAL_ODDS_LOOKUP = {
    ("cornershc", 0.5): {"over": 1.990, "under": 1.850},
    ("cornershc", 1.5): {"over": 2.033, "under": 1.737},
    ("cornershc", 2.5): {"over": 2.200, "under": 1.620},
    ("cardshc", 0.5): {"over": 1.960, "under": 1.790},
}


def estimate_market_odds(
    market: str,
    features_df: pd.DataFrame,
    margin: float = 0.05,
) -> float:
    """Estimate fair odds for a niche market from historical base rates.

    Computes the empirical probability of the outcome from the features dataset,
    then converts to decimal odds with a bookmaker margin.

    Args:
        market: Market name, e.g. 'cards_under_35', 'corners_over_85'.
        features_df: Full features DataFrame with outcome columns.
        margin: Bookmaker overround (default 5%).

    Returns:
        Estimated fair decimal odds (e.g. 1.66 for corners_over_85).
    """
    stat, direction, line = parse_market_name(market)

    if line is None:
        return 2.50  # H2H / base markets — keep flat fallback

    # HC markets: use real median odds from historical collection
    if stat in _HC_STAT_CONFIG:
        lookup_key = (stat, line)
        real_odds = _HC_REAL_ODDS_LOOKUP.get(lookup_key)
        if real_odds is None:
            logger.warning(f"No HC odds lookup for {market} (line={line})")
            return 2.50
        fair_odds = real_odds[direction]
        logger.info(
            f"  {market}: HC lookup ±{line} {direction}={fair_odds:.3f} "
            f"(vs flat 2.50, diff={((2.50 / fair_odds) - 1) * 100:+.0f}%)"
        )
        return fair_odds

    # Totals markets: compute from base rate
    if stat in _STAT_CONFIG:
        total_col = _STAT_CONFIG[stat][0]
        if total_col not in features_df.columns:
            logger.warning(f"Column {total_col} not found for market {market}")
            return 2.50
        values = features_df[total_col].dropna()
    else:
        logger.warning(f"Unknown stat {stat} for market {market}")
        return 2.50

    # Compute base rate
    if direction == "over":
        base_rate = (values > line).mean()
    else:
        base_rate = (values < line).mean()

    if base_rate <= 0 or base_rate >= 1:
        logger.warning(f"Degenerate base rate {base_rate:.4f} for {market}")
        return 2.50

    fair_odds = (1 + margin) / base_rate
    logger.info(
        f"  {market}: base_rate={base_rate:.3f}, "
        f"fair_odds={fair_odds:.2f} (vs flat 2.50, diff={((2.50 / fair_odds) - 1) * 100:+.0f}%)"
    )
    return fair_odds


def estimate_per_match_odds(
    holdout_df: pd.DataFrame,
    features_df: pd.DataFrame,
    margin: float = 0.05,
) -> pd.DataFrame:
    """Replace flat 2.50 fallback odds with per-match estimated odds.

    For each niche market bet in holdout_df, estimates fair odds using:
    1. Global base rate from the features dataset
    2. Per-match EMA adjustment (team-specific corner/card/shot/foul rates vs league avg)

    Args:
        holdout_df: Holdout predictions with columns: fixture_id, market, odds.
        features_df: Features parquet with fixture_id, EMA columns, league averages.
        margin: Bookmaker overround (default 5%).

    Returns:
        holdout_df with odds replaced where flat fallback was used.
    """
    df = holdout_df.copy()

    # Pre-compute global base rates per market
    markets = df["market"].unique()
    global_rates: Dict[str, float] = {}
    for market in markets:
        stat, direction, line = parse_market_name(market)
        if line is None:
            continue
        if stat in _STAT_CONFIG:
            total_col = _STAT_CONFIG[stat][0]
            if total_col not in features_df.columns:
                continue
            values = features_df[total_col].dropna()
        elif stat in _HC_STAT_CONFIG:
            home_col, away_col = _HC_STAT_CONFIG[stat][:2]
            if home_col not in features_df.columns or away_col not in features_df.columns:
                continue
            values = (features_df[home_col] - features_df[away_col]).dropna()
        else:
            continue
        if direction == "over":
            global_rates[market] = (values > line).mean()
        else:
            global_rates[market] = (values < line).mean()

    # Merge EMA columns from features into holdout for per-match adjustment
    ema_cols = set()
    for stat, (_, h_ema, a_ema, lg_avg) in _STAT_CONFIG.items():
        ema_cols.update([h_ema, a_ema, lg_avg])
    available_ema = [c for c in ema_cols if c in features_df.columns]

    if available_ema and "fixture_id" in features_df.columns:
        feat_subset = features_df[["fixture_id"] + available_ema].drop_duplicates("fixture_id")
        df = df.merge(feat_subset, on="fixture_id", how="left")
    else:
        logger.warning("Cannot merge EMA columns — using global base rates only")

    replaced = 0
    flat_odds_value = 2.50

    for idx, row in df.iterrows():
        market = row["market"]
        stat, direction, line = parse_market_name(market)

        if line is None or (stat not in _STAT_CONFIG and stat not in _HC_STAT_CONFIG):
            continue

        # Only replace flat 2.50 fallback odds
        if abs(row["odds"] - flat_odds_value) > 0.01:
            continue

        # HC markets: use real median odds from historical collection
        if stat in _HC_STAT_CONFIG:
            lookup_key = (stat, line)
            real_odds = _HC_REAL_ODDS_LOOKUP.get(lookup_key)
            if real_odds is None:
                continue
            estimated_odds = real_odds[direction]
            df.at[idx, "odds"] = estimated_odds
            replaced += 1
            continue

        base_rate = global_rates.get(market)
        if base_rate is None or base_rate <= 0 or base_rate >= 1:
            continue

        # Per-match EMA adjustment (totals markets only)
        ema_factor = 1.0
        _, h_ema_col, a_ema_col, lg_avg_col = _STAT_CONFIG[stat]
        if h_ema_col in df.columns and a_ema_col in df.columns and lg_avg_col in df.columns:
            h_ema = row.get(h_ema_col)
            a_ema = row.get(a_ema_col)
            lg_avg = row.get(lg_avg_col)
            if pd.notna(h_ema) and pd.notna(a_ema) and pd.notna(lg_avg) and lg_avg > 0:
                match_expected = h_ema + a_ema
                ema_factor = match_expected / lg_avg
                ema_factor = np.clip(ema_factor, 0.8, 1.2)

        # Adjust base rate with EMA factor
        if direction == "over":
            adj_rate = base_rate * ema_factor
        else:
            adj_rate = base_rate / ema_factor

        adj_rate = np.clip(adj_rate, 0.01, 0.99)
        estimated_odds = (1 + margin) / adj_rate

        df.at[idx, "odds"] = estimated_odds
        replaced += 1

    # Clean up merged EMA columns
    for col in available_ema:
        if col in df.columns and col not in holdout_df.columns:
            df.drop(columns=col, inplace=True)

    logger.info(f"Replaced {replaced} flat 2.50 odds with estimated odds")

    # Log summary per market
    for market in sorted(global_rates.keys()):
        rate = global_rates[market]
        global_fair = (1 + margin) / rate
        n_market = ((holdout_df["market"] == market) & (abs(holdout_df["odds"] - flat_odds_value) < 0.01)).sum()
        logger.info(
            f"  {market}: global_rate={rate:.3f}, global_fair={global_fair:.2f}, "
            f"flat_2.50_bets={n_market}"
        )

    return df


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


def _build_fixture_team_map(holdout_dir: Path) -> Dict[int, Dict]:
    """Map fixture_id → {date, home_team, away_team, league} from raw match data.

    Loads matches.parquet from data/01-raw/{league}/{season}/ for all leagues/seasons.
    """
    raw_root = Path("data/01-raw")
    fixture_map: Dict[int, Dict] = {}

    if not raw_root.exists():
        logger.warning(f"Raw data root not found: {raw_root}")
        return fixture_map

    for league_dir in sorted(raw_root.iterdir()):
        if not league_dir.is_dir():
            continue
        league = league_dir.name
        for season_dir in sorted(league_dir.iterdir()):
            matches_path = season_dir / "matches.parquet"
            if not matches_path.exists():
                continue
            try:
                df = pd.read_parquet(
                    matches_path,
                    columns=[
                        "fixture.id",
                        "fixture.date",
                        "teams.home.name",
                        "teams.away.name",
                    ],
                )
                for _, row in df.iterrows():
                    fid = int(row["fixture.id"])
                    date_str = str(row["fixture.date"])[:10]
                    fixture_map[fid] = {
                        "date": date_str,
                        "home_team": row["teams.home.name"],
                        "away_team": row["teams.away.name"],
                        "league": league,
                    }
            except Exception as e:
                logger.debug(f"Skipping {matches_path}: {e}")

    logger.info(f"Built fixture map: {len(fixture_map)} fixtures from raw data")
    return fixture_map


def _fuzzy_team_match(name_a: str, name_b: str, threshold: int = 75) -> bool:
    """Check if two team names match using fuzzy matching.

    Uses max of ratio, partial_ratio, and token_sort_ratio to handle
    naming variations like 'Bayern München' vs 'Bayern Munich',
    'Lens' vs 'RC Lens', 'Stade Brestois 29' vs 'Brest'.
    """
    a = name_a.lower().strip()
    b = name_b.lower().strip()
    if a == b:
        return True
    if a in b or b in a:
        return True
    score = max(
        fuzz.ratio(a, b),
        fuzz.partial_ratio(a, b),
        fuzz.token_sort_ratio(a, b),
    )
    return score >= threshold


def merge_historical_odds(
    df: pd.DataFrame,
    historical_odds_path: str,
) -> pd.DataFrame:
    """Replace flat fallback odds in holdout predictions with real historical odds.

    Args:
        df: Holdout predictions DataFrame with columns: date, fixture_id, market, odds.
        historical_odds_path: Path to niche_odds_historical.parquet.

    Returns:
        DataFrame with odds replaced where historical data matches.
    """
    hist_path = Path(historical_odds_path)
    if not hist_path.exists():
        logger.warning(f"Historical odds file not found: {hist_path}")
        return df

    hist_df = pd.read_parquet(hist_path)
    logger.info(f"Loaded {len(hist_df)} historical odds records")

    if hist_df.empty:
        return df

    df = df.copy()

    # Build fixture → team name mapping from raw data
    fixture_map = _build_fixture_team_map(Path("data/01-raw"))

    # Track match rates per market
    match_counts: Dict[str, int] = {}
    total_counts: Dict[str, int] = {}
    replaced_count = 0

    for idx, row in df.iterrows():
        market = row["market"]

        # Parse market name to get stat, direction, line
        stat, direction, line = parse_market_name(market)
        if stat not in ("cards", "corners", "cornershc", "cardshc") or line is None:
            continue

        total_counts[market] = total_counts.get(market, 0) + 1

        # Get team names for this fixture
        fid = int(row["fixture_id"])
        fixture_info = fixture_map.get(fid)
        if fixture_info is None:
            continue

        date_str = fixture_info["date"]
        home_team = fixture_info["home_team"]
        away_team = fixture_info["away_team"]
        league = fixture_info["league"]

        # Find matching historical odds
        # Filter by date, league, market (stat), and line
        candidates = hist_df[
            (hist_df["date"] == date_str)
            & (hist_df["league"] == league)
            & (hist_df["market"] == stat)
            & (hist_df["line"].between(line - 0.01, line + 0.01))
        ]

        if candidates.empty:
            continue

        # Fuzzy match on team names
        matched = False
        for _, hrow in candidates.iterrows():
            if _fuzzy_team_match(home_team, hrow["home_team"]) and _fuzzy_team_match(
                away_team, hrow["away_team"]
            ):
                # Replace odds
                odds_col = f"{direction}_avg"
                new_odds = hrow.get(odds_col)
                if new_odds is not None and pd.notna(new_odds) and new_odds > 1.0:
                    old_odds = df.at[idx, "odds"]
                    df.at[idx, "odds"] = new_odds
                    replaced_count += 1
                    match_counts[market] = match_counts.get(market, 0) + 1
                    matched = True
                break

    # Log match rates
    logger.info(f"\n--- Historical Odds Match Rates ---")
    logger.info(f"Total replacements: {replaced_count}")
    for market in sorted(total_counts.keys()):
        total = total_counts[market]
        matched = match_counts.get(market, 0)
        pct = matched / total * 100 if total > 0 else 0
        logger.info(f"  {market}: {matched}/{total} matched ({pct:.1f}%)")

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
    parser.add_argument(
        "--historical-odds",
        type=str,
        default=None,
        help="Path to historical niche odds parquet (replaces flat 2.50 fallback)",
    )
    parser.add_argument(
        "--estimated-odds",
        action="store_true",
        help="Estimate fair odds from base rates (replaces flat 2.50 for remaining bets)",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="data/03-features/features_all_5leagues_with_odds.parquet",
        help="Path to features parquet for base rate estimation",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Bookmaker margin for estimated odds (default 5%%)",
    )

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

    if args.historical_odds:
        logger.info(f"Merging historical odds from {args.historical_odds}...")
        df = merge_historical_odds(df, args.historical_odds)

    if args.estimated_odds:
        features_path = Path(args.features_path)
        if not features_path.exists():
            logger.error(f"Features file not found: {features_path}")
            sys.exit(1)
        logger.info(f"Loading features from {features_path} for odds estimation...")
        features_df = pd.read_parquet(features_path)

        # Print global base rate summary
        logger.info("--- Global Base Rate Odds Estimates ---")
        for market in sorted(df["market"].unique()):
            estimate_market_odds(market, features_df, margin=args.margin)

        # Replace flat 2.50 with per-match estimated odds
        logger.info("Replacing flat 2.50 odds with per-match estimated odds...")
        df = estimate_per_match_odds(df, features_df, margin=args.margin)

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
