"""
Centralized league ID mapping for API-Football.

Single source of truth — all modules should import from here.
"""
from typing import Dict

# API-Football league IDs
LEAGUE_IDS: Dict[str, int] = {
    # Tier 1: European Big 5
    "premier_league": 39,
    "la_liga": 140,
    "serie_a": 135,
    "bundesliga": 78,
    "ligue_1": 61,
    # Tier 1.5: European secondary
    "ekstraklasa": 106,
    # Tier 2: Americas
    "mls": 253,
    "liga_mx": 262,
}

# League groups for separate model pools
EUROPEAN_LEAGUES = [
    "premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1", "ekstraklasa",
]
AMERICAS_LEAGUES = [
    "mls", "liga_mx",
]
