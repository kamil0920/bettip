"""
Shared line plausibility utilities for niche market filtering.

Used by both training (sniper optimization, feature param optimization)
and inference (daily recommendations, match scheduler) to ensure
consistent plausibility logic.

Training filter: Exclude rows from leagues where the bet line falls
outside the league's typical range (expanding mean ± buffer).

Inference filter: Skip lines bookmakers unlikely offer for a league.
"""
import logging
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bookmaker line plausibility: lines are typically within ±buffer of league average
# Buffer varies by market (fouls/shots have more variance than corners/cards)
LINE_PLAUSIBILITY = {
    "fouls": {"stat": "total_fouls", "buffer": 4.0},
    "shots": {"stat": "total_shots", "buffer": 4.0},
    "cards": {"stat": "total_cards", "buffer": 2.0},
    "corners": {"stat": "total_corners", "buffer": 2.0},
    "goals": {"stat": "total_goals", "buffer": 2.0},
    "hgoals": {"stat": "home_goals", "buffer": 1.5},
    "agoals": {"stat": "away_goals", "buffer": 1.5},
    "cornershc": {"stat": "corner_diff", "buffer": 3.0},
    "cardshc": {"stat": "card_diff", "buffer": 2.0},
}


def parse_market_line(market_name: str) -> Optional[Tuple[str, float, str]]:
    """Parse a niche line variant market name into components.

    Args:
        market_name: e.g. 'corners_over_85', 'fouls_under_255'

    Returns:
        Tuple of (base_market, target_line, direction) or None for non-line markets.
        Example: ('corners', 8.5, 'over')
    """
    m = re.match(r"^(corners|shots|fouls|cards|goals|hgoals|agoals|cornershc|cardshc)_(over|under)_(\d+)$", market_name)
    if not m:
        return None
    return m.group(1), float(m.group(3)) / 10.0, m.group(2)


def compute_league_stat_averages(
    df: pd.DataFrame,
    stat_cols: Optional[list] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-league stat averages from a features DataFrame.

    Args:
        df: DataFrame with 'league' column and stat columns.
        stat_cols: Stat columns to compute averages for.
            Defaults to all stats in LINE_PLAUSIBILITY.

    Returns:
        Dict like {"eredivisie": {"total_fouls": 20.3, ...}, ...}
    """
    if "league" not in df.columns:
        return {}

    if stat_cols is None:
        stat_cols = [cfg["stat"] for cfg in LINE_PLAUSIBILITY.values()]

    available_cols = [c for c in stat_cols if c in df.columns]
    if not available_cols:
        return {}

    result: Dict[str, Dict[str, float]] = {}
    for league, group in df.groupby("league"):
        result[league] = {}
        for col in available_cols:
            avg = group[col].mean()
            if pd.notna(avg):
                result[league][col] = float(avg)

    return result


def check_line_plausible(
    market_name: str,
    league: str,
    league_stats: Dict[str, Dict[str, float]],
    odds_row: Optional[pd.Series] = None,
) -> Tuple[str, str]:
    """Check if a niche market line is plausible for this league.

    Args:
        market_name: Full market name (e.g. 'fouls_over_255')
        league: League key (e.g. 'eredivisie')
        league_stats: Per-league stat averages from compute_league_stat_averages()
        odds_row: Optional odds row for corners available_lines check

    Returns:
        Tuple of (status, reason) where status is 'yes' / 'no' / 'unknown'
    """
    parsed = parse_market_line(market_name)
    if parsed is None:
        return "yes", "non-line market"

    base_market, target_line, direction = parsed
    config = LINE_PLAUSIBILITY.get(base_market)
    if not config:
        return "unknown", "no plausibility config"

    # For corners: use real available_lines from The Odds API if provided
    if base_market == "corners" and odds_row is not None:
        avail = odds_row.get("corners_available_lines")
        if avail is not None and not (isinstance(avail, float) and pd.isna(avail)):
            try:
                available_lines = sorted(float(x) for x in avail)
                if target_line in available_lines:
                    return "yes", f"line in bookmaker list {available_lines}"
                return "no", f"line {target_line} not in {available_lines}"
            except (TypeError, ValueError):
                pass

    stat_name = config["stat"]
    buffer = config["buffer"]

    league_data = league_stats.get(league)
    if not league_data:
        return "unknown", f"no stats for league '{league}'"

    league_avg = league_data.get(stat_name)
    if league_avg is None:
        return "unknown", f"no {stat_name} for league '{league}'"

    low = league_avg - buffer
    high = league_avg + buffer
    if low <= target_line <= high:
        return "yes", f"line {target_line} within [{low:.1f}, {high:.1f}] (avg={league_avg:.1f})"
    return "no", f"line {target_line} outside [{low:.1f}, {high:.1f}] (avg={league_avg:.1f})"


def filter_implausible_training_rows(
    df: pd.DataFrame,
    bet_type: str,
) -> pd.DataFrame:
    """Filter training rows from leagues where the bet line is implausible.

    For niche line variant markets (e.g. fouls_over_255), excludes rows from
    leagues where the line falls outside `league_expanding_avg ± buffer`.
    Uses per-league expanding mean (shifted by 1) to prevent data leakage.

    For non-line markets, returns df unchanged.

    Args:
        df: Training DataFrame with 'league' column and stat columns.
        bet_type: Market name (e.g. 'fouls_over_255')

    Returns:
        Filtered DataFrame (or original if not a line variant market)
    """
    parsed = parse_market_line(bet_type)
    if parsed is None:
        return df

    base_market, target_line, direction = parsed
    config = LINE_PLAUSIBILITY.get(base_market)
    if not config:
        return df

    stat_col = config["stat"]
    buffer = config["buffer"]

    if stat_col not in df.columns or "league" not in df.columns:
        logger.warning(
            f"Cannot filter: missing '{stat_col}' or 'league' column"
        )
        return df

    n_before = len(df)

    # Compute per-league expanding mean with shift(1) to prevent leakage
    df = df.copy()
    df["_league_expanding_avg"] = df.groupby("league")[stat_col].transform(
        lambda x: x.shift(1).expanding(min_periods=10).mean()
    )

    # Keep rows where the league expanding avg makes the line plausible
    low = df["_league_expanding_avg"] - buffer
    high = df["_league_expanding_avg"] + buffer
    mask = (low <= target_line) & (target_line <= high)

    # Also keep rows where expanding avg is NaN (insufficient history)
    mask = mask | df["_league_expanding_avg"].isna()

    # Log per-league filtering
    for league in df["league"].unique():
        league_mask = df["league"] == league
        league_total = league_mask.sum()
        league_kept = (league_mask & mask).sum()
        league_removed = league_total - league_kept
        if league_removed > 0:
            league_avg = df.loc[league_mask, "_league_expanding_avg"].median()
            logger.info(
                f"Filtered {league_removed}/{league_total} {league} rows "
                f"for {bet_type}: league avg {league_avg:.1f} "
                f"outside [{target_line - buffer:.1f}, {target_line + buffer:.1f}]"
            )

    df = df[mask].drop(columns=["_league_expanding_avg"]).reset_index(drop=True)

    n_after = len(df)
    if n_before != n_after:
        logger.info(
            f"Training plausibility filter: {n_before} -> {n_after} rows "
            f"({n_before - n_after} removed for {bet_type})"
        )

    return df
