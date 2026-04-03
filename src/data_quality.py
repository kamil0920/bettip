"""Data quality utilities for blocking unreliable league-market predictions."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Mapping from market name patterns to base market families
_BASE_MARKET_PATTERNS = [
    (re.compile(r"^cards"), "cards"),
    (re.compile(r"^cardshc"), "cards"),
    (re.compile(r"^fouls"), "fouls"),
    (re.compile(r"^corners"), "corners"),
    (re.compile(r"^cornershc"), "corners"),
    (re.compile(r"^shots"), "shots"),
    (re.compile(r"^btts"), "btts"),
    (re.compile(r"^(over|under)\d"), "goals"),
    (re.compile(r"^(home_win|away_win)"), "h2h"),
    (re.compile(r"^ht"), "ht"),
]


def get_base_market(market_name: str) -> str:
    """Extract base market family from a market name.

    Examples:
        cards_under_25 -> cards
        fouls_over_225 -> fouls
        corners_over_85 -> corners
        cardshc_under_05 -> cards
        over25 -> goals
        home_win -> h2h
    """
    name = market_name.lower().strip()
    for pattern, family in _BASE_MARKET_PATTERNS:
        if pattern.match(name):
            return family
    return name


def load_blocklist(
    strategies_path: str = "config/strategies.yaml",
) -> Dict[str, List[str]]:
    """Load data quality blocklist from strategies config.

    Returns:
        Dict mapping base market -> list of blocked leagues.
        Empty dict if no blocklist configured.
    """
    path = Path(strategies_path)
    if not path.exists():
        return {}

    with open(path) as f:
        config = yaml.safe_load(f)

    dq = config.get("data_quality", {})
    return dq.get("blocklist", {})


def load_filter_mode_overrides(
    strategies_path: str = "config/strategies.yaml",
) -> Dict[str, str]:
    """Load per-market filter mode overrides from strategies config.

    Returns:
        Dict mapping base market -> filter mode ('blocklist', 'nan_filter', 'hybrid').
        Empty dict if no overrides configured.
    """
    path = Path(strategies_path)
    if not path.exists():
        return {}

    with open(path) as f:
        config = yaml.safe_load(f)

    dq = config.get("data_quality", {})
    overrides = dq.get("filter_mode_overrides", {})
    valid_modes = ("blocklist", "nan_filter", "hybrid")
    return {k: v for k, v in overrides.items() if v in valid_modes}


def is_market_blocked_for_league(
    market_name: str,
    league: str,
    blocklist: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """Check if a market is blocked for a specific league.

    Args:
        market_name: Full market name (e.g. 'cards_under_25').
        league: League slug (e.g. 'eredivisie').
        blocklist: Pre-loaded blocklist dict. If None, loads from default config.

    Returns:
        True if the market should be blocked for this league.
    """
    if blocklist is None:
        blocklist = load_blocklist()

    if not blocklist:
        return False

    base = get_base_market(market_name)
    blocked_leagues = blocklist.get(base, [])
    return league.lower().strip() in blocked_leagues


def load_inactive_leagues(
    config_path: str = "config/inactive_leagues.yaml",
) -> List[str]:
    """Load list of inactive leagues to exclude from training.

    Returns:
        List of league slugs. Empty list if config missing.
    """
    path = Path(config_path)
    if not path.exists():
        return []

    with open(path) as f:
        config = yaml.safe_load(f)

    return config.get("inactive_leagues", [])


def fix_fake_zero_cards(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and NaN-ify fake zero card data from API-Football.

    API-Football often returns match results without card events, causing
    cards to be recorded as 0 when data is actually missing. This corrupts
    card market models (inflates UNDER targets, deflates OVER targets).

    Detection rule: cards=0 with fouls>3 is almost certainly missing data,
    not a real zero-card match. Real 0-card matches are extremely rare (<0.3%).

    Returns:
        DataFrame with fake card zeros replaced by NaN.
    """
    if "home_fouls" not in df.columns:
        return df

    df = df.copy()
    raw_card_cols = [
        c for c in ["home_cards", "away_cards", "total_cards",
                     "home_yellow_cards", "away_yellow_cards",
                     "home_red_cards", "away_red_cards"]
        if c in df.columns
    ]
    if not raw_card_cols:
        return df

    n_before = sum(
        (df[c] == 0).sum() for c in raw_card_cols if c in df.columns
    )

    # Both sides zero + fouls indicate missing data
    both_zero = (
        df.get("home_cards", pd.Series(dtype=float)).eq(0)
        & df.get("away_cards", pd.Series(dtype=float)).eq(0)
        & ((df["home_fouls"] > 3) | (df["away_fouls"] > 3))
    )
    for col in raw_card_cols:
        df.loc[both_zero, col] = np.nan

    # Per-side: yellow_cards=0 with fouls>5 on same side
    home_cols = [c for c in raw_card_cols if "home" in c]
    away_cols = [c for c in raw_card_cols if "away" in c]

    home_fake = (
        df.get("home_yellow_cards", pd.Series(dtype=float)).eq(0)
        & (df["home_fouls"] > 5)
        & ~both_zero
    )
    for col in home_cols:
        df.loc[home_fake, col] = np.nan

    away_fake = (
        df.get("away_yellow_cards", pd.Series(dtype=float)).eq(0)
        & (df["away_fouls"] > 5)
        & ~both_zero
    )
    for col in away_cols:
        df.loc[away_fake, col] = np.nan

    # Away-only fake zeros: away_cards==0 when home_cards>0 (19% of matches).
    # API-Football returns 0 instead of NaN for missing away card data.
    # home_cards is never 0 in production (min=1), so away_cards==0 with
    # home_cards>0 is the fake-zero signature.
    if "away_cards" in df.columns and "home_cards" in df.columns:
        away_only_fake = (
            (df["away_cards"] == 0)
            & (df["home_cards"] > 0)
            & ~both_zero  # Already handled above
        )
        for col in away_cols:
            df.loc[away_only_fake, col] = np.nan
        n_away_fake = away_only_fake.sum()
    else:
        n_away_fake = 0

    # Recompute total_cards
    if "total_cards" in df.columns and "home_cards" in df.columns:
        df["total_cards"] = df["home_cards"] + df["away_cards"]

    n_fixed = both_zero.sum() + home_fake.sum() + away_fake.sum() + n_away_fake
    logger.info(
        f"fix_fake_zero_cards: {n_fixed} matches had fake zero cards → NaN "
        f"(both_zero={both_zero.sum()}, home={home_fake.sum()}, "
        f"away_yellow={away_fake.sum()}, away_only={n_away_fake})"
    )
    return df


def fix_fake_zero_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and NaN-ify fake zero shots/fouls data from API-Football.

    API-Football returns 0 instead of NaN for missing match statistics.
    A match with goals but 0 shots is physically impossible.
    Affects ~130 Belgian Pro League + 4 Scottish Premiership matches.

    Also fixes fake-zero away_corners (3.1% rate vs 1.5% for home_corners).

    Returns:
        DataFrame with fake stat zeros replaced by NaN.
    """
    df = df.copy()
    n_fixed = 0

    # --- Shots & fouls: both zero with goals scored = impossible ---
    has_goals = df.get("total_goals", pd.Series(dtype=float)).fillna(0) > 0
    if "total_shots" not in has_goals.index.__class__.__name__:
        # Derive total_goals from ft_home/ft_away if needed
        if "total_goals" not in df.columns and "ft_home" in df.columns:
            has_goals = (df["ft_home"].fillna(0) + df["ft_away"].fillna(0)) > 0

    shots_cols = [
        c for c in ["home_shots", "away_shots", "total_shots",
                     "home_shots_on_target", "away_shots_on_target",
                     "total_shots_on_target"]
        if c in df.columns
    ]
    fouls_cols = [
        c for c in ["home_fouls", "away_fouls", "total_fouls"]
        if c in df.columns
    ]

    # Fake shots: total_shots==0 with goals scored
    if "total_shots" in df.columns:
        fake_shots = (df["total_shots"] == 0) & has_goals
        for col in shots_cols:
            df.loc[fake_shots, col] = np.nan
        n_shots = fake_shots.sum()
        if n_shots:
            logger.info(f"fix_fake_zero_stats: {n_shots} matches with fake zero shots → NaN")
        n_fixed += n_shots

    # Fake fouls: home_fouls==0 AND away_fouls==0 with goals scored
    if "home_fouls" in df.columns and "away_fouls" in df.columns:
        fake_fouls = (df["home_fouls"] == 0) & (df["away_fouls"] == 0) & has_goals
        for col in fouls_cols:
            df.loc[fake_fouls, col] = np.nan
        n_fouls = fake_fouls.sum()
        if n_fouls:
            logger.info(f"fix_fake_zero_stats: {n_fouls} matches with fake zero fouls → NaN")
        n_fixed += n_fouls

    # --- Away corners: 0 when home_corners>0 and total==home (same API pattern as cards) ---
    if all(c in df.columns for c in ["away_corners", "home_corners", "total_corners"]):
        fake_away_corners = (
            (df["away_corners"] == 0)
            & (df["home_corners"] > 0)
            & (df["total_corners"] == df["home_corners"])
        )
        df.loc[fake_away_corners, "away_corners"] = np.nan
        df.loc[fake_away_corners, "total_corners"] = np.nan  # Now invalid
        n_corners = fake_away_corners.sum()
        if n_corners:
            logger.info(f"fix_fake_zero_stats: {n_corners} matches with fake zero away_corners → NaN")
        n_fixed += n_corners

    logger.info(f"fix_fake_zero_stats: {n_fixed} total fake zeros fixed")
    return df


def fix_corrupted_odds(df: pd.DataFrame) -> pd.DataFrame:
    """NaN-ify corrupted odds rows where avg_*_close is clearly wrong.

    Audit found 2 La Liga 2 fixtures (1217725, 1217729) from 2024-12-21
    with avg_home_close wildly diverging from b365 (3.59 vs 1.62) and
    max_home_close >29 (garbage outlier bookmaker in the average).

    Detection: overround (1/H + 1/D + 1/A) < 0.90 = corrupted average.
    """
    avg_cols = ["avg_home_close", "avg_draw_close", "avg_away_close"]
    if not all(c in df.columns for c in avg_cols):
        return df

    df = df.copy()
    valid = df[avg_cols].notna().all(axis=1)
    overround = (1 / df["avg_home_close"] + 1 / df["avg_draw_close"] + 1 / df["avg_away_close"])

    corrupted = valid & (overround < 0.90)
    n_corrupted = corrupted.sum()
    if n_corrupted:
        # NaN all avg/max odds for these rows — b365 may still be OK
        for c in df.columns:
            if c.startswith("avg_") or c.startswith("max_"):
                df.loc[corrupted, c] = np.nan
        # Also NaN derived odds features
        for c in df.columns:
            if c.startswith("odds_") and c != "odds_upset_potential":
                df.loc[corrupted, c] = np.nan
        logger.info(f"fix_corrupted_odds: {n_corrupted} rows with overround < 0.90 → NaN")

    return df
