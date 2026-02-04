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
    # Tier 1 Expansion: European
    "eredivisie": 88,
    "portuguese_liga": 94,
    "turkish_super_lig": 203,
    "belgian_pro_league": 144,
    "scottish_premiership": 179,
    # Tier 2: Americas
    "mls": 253,
    "liga_mx": 262,
}

# League groups for separate model pools
TIER1_EXPANSION = [
    "eredivisie", "portuguese_liga", "turkish_super_lig",
    "belgian_pro_league", "scottish_premiership",
]
EUROPEAN_LEAGUES = [
    "premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1",
    "ekstraklasa",
] + TIER1_EXPANSION
AMERICAS_LEAGUES = [
    "mls", "liga_mx",
]
