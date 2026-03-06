"""Data quality utilities for blocking unreliable league-market predictions."""

import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

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
