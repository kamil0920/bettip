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

    # Recompute total_cards
    if "total_cards" in df.columns and "home_cards" in df.columns:
        df["total_cards"] = df["home_cards"] + df["away_cards"]

    n_fixed = both_zero.sum() + home_fake.sum() + away_fake.sum()
    logger.info(
        f"fix_fake_zero_cards: {n_fixed} matches had fake zero cards → NaN"
    )
    return df
