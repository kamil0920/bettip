"""
API-Football Niche Odds Loader

Fetches niche market odds from API-Football's /odds endpoint:
- Bet ID 8:  Both Teams Score (BTTS)
- Bet ID 45: Corners Over/Under
- Bet ID 80: Cards Over/Under
- Bet ID 87: Total Shots On Goal

Uses the existing FootballAPIClient for rate limiting, auth, and retry handling.

Usage:
    loader = ApiFootballOddsLoader()
    df = loader.fetch_niche_odds("premier_league", fixture_ids=[1379200, 1379201])
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_collection.api_client import FootballAPIClient

logger = logging.getLogger(__name__)

LEAGUE_IDS: Dict[str, int] = {
    "premier_league": 39,
    "la_liga": 140,
    "serie_a": 135,
    "bundesliga": 78,
    "ligue_1": 61,
    "ekstraklasa": 106,
}

# API-Football bet type IDs for niche markets
BET_IDS: Dict[str, int] = {
    "btts": 8,
    "corners": 45,
    "cards": 80,
    "shots_ot": 87,
}

# Mapping from bet key to value parsing rules
# Each entry: (yes_value_pattern, no_value_pattern, has_line)
BET_PARSERS: Dict[str, Dict[str, Any]] = {
    "btts": {
        "yes_pattern": "Yes",
        "no_pattern": "No",
        "has_line": False,
    },
    "corners": {
        "over_pattern": r"Over\s+([\d.]+)",
        "under_pattern": r"Under\s+([\d.]+)",
        "has_line": True,
    },
    "cards": {
        "over_pattern": r"Over\s+([\d.]+)",
        "under_pattern": r"Under\s+([\d.]+)",
        "has_line": True,
    },
    "shots_ot": {
        "over_pattern": r"Over\s+([\d.]+)",
        "under_pattern": r"Under\s+([\d.]+)",
        "has_line": True,
    },
}


def _parse_line_from_value(value: str) -> Optional[float]:
    """Extract numeric line from value string like 'Over 8.5'."""
    match = re.search(r"([\d.]+)", value)
    if match:
        return float(match.group(1))
    return None


class ApiFootballOddsLoader:
    """Loads niche market odds from API-Football /odds endpoint."""

    def __init__(self, cache_dir: str = "data/niche_odds_cache") -> None:
        self.client = FootballAPIClient()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_upcoming_fixture_ids(
        self, league: str, next_n: int = 20
    ) -> List[int]:
        """Get fixture IDs for upcoming matches in a league."""
        league_id = LEAGUE_IDS.get(league)
        if league_id is None:
            logger.warning(f"Unknown league: {league}")
            return []

        response = self.client._make_request(
            "/fixtures", {"league": league_id, "next": next_n}
        )
        fixtures = response.get("response", [])
        ids = [f["fixture"]["id"] for f in fixtures]
        logger.info(f"{league}: found {len(ids)} upcoming fixtures")
        return ids

    def _fetch_odds_for_fixture(
        self, fixture_id: int, bet_id: int
    ) -> List[Dict[str, Any]]:
        """Fetch odds for a single fixture and bet type."""
        response = self.client._make_request(
            "/odds", {"fixture": fixture_id, "bet": bet_id}
        )
        return response.get("response", [])

    def _parse_btts_odds(
        self, bookmakers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse BTTS (Yes/No) odds from bookmaker data."""
        yes_odds: List[float] = []
        no_odds: List[float] = []

        for bm in bookmakers:
            for bet in bm.get("bets", []):
                for val in bet.get("values", []):
                    odd = float(val["odd"])
                    if val["value"] == "Yes":
                        yes_odds.append(odd)
                    elif val["value"] == "No":
                        no_odds.append(odd)

        result: Dict[str, Any] = {}
        if yes_odds:
            result["niche_btts_yes_avg"] = np.mean(yes_odds)
            result["niche_btts_yes_max"] = np.max(yes_odds)
        if no_odds:
            result["niche_btts_no_avg"] = np.mean(no_odds)
            result["niche_btts_no_max"] = np.max(no_odds)
        return result

    def _parse_over_under_odds(
        self, bookmakers: List[Dict[str, Any]], prefix: str
    ) -> Dict[str, Any]:
        """Parse Over/Under odds with line extraction."""
        over_odds: List[float] = []
        under_odds: List[float] = []
        lines: List[float] = []

        for bm in bookmakers:
            for bet in bm.get("bets", []):
                for val in bet.get("values", []):
                    odd = float(val["odd"])
                    value_str = val["value"]
                    line = _parse_line_from_value(value_str)

                    if value_str.startswith("Over"):
                        over_odds.append(odd)
                        if line is not None:
                            lines.append(line)
                    elif value_str.startswith("Under"):
                        under_odds.append(odd)
                        if line is not None and not lines:
                            lines.append(line)

        result: Dict[str, Any] = {}
        if over_odds:
            result[f"niche_{prefix}_over_avg"] = np.mean(over_odds)
            result[f"niche_{prefix}_over_max"] = np.max(over_odds)
        if under_odds:
            result[f"niche_{prefix}_under_avg"] = np.mean(under_odds)
            result[f"niche_{prefix}_under_max"] = np.max(under_odds)
        if lines:
            # Use the most common line (mode)
            from collections import Counter

            line_counts = Counter(lines)
            result[f"niche_{prefix}_line"] = line_counts.most_common(1)[0][0]
        return result

    def fetch_niche_odds(
        self, league: str, fixture_ids: List[int]
    ) -> pd.DataFrame:
        """Fetch niche odds for a list of fixtures.

        Args:
            league: League identifier (e.g. "premier_league").
            fixture_ids: List of API-Football fixture IDs.

        Returns:
            DataFrame with one row per fixture and niche odds columns.
        """
        rows: List[Dict[str, Any]] = []

        for fid in fixture_ids:
            row: Dict[str, Any] = {"fixture_id": fid}
            n_bookmakers = 0

            for market, bet_id in BET_IDS.items():
                try:
                    data = self._fetch_odds_for_fixture(fid, bet_id)
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch {market} odds for fixture {fid}: {e}"
                    )
                    continue

                if not data:
                    continue

                bookmakers = data[0].get("bookmakers", [])
                n_bookmakers = max(n_bookmakers, len(bookmakers))

                # Extract fixture info from first response
                if "home_team" not in row:
                    fixture_info = data[0].get("fixture", {})
                    league_info = data[0].get("league", {})
                    teams = data[0].get("teams", {}) if "teams" in data[0] else {}
                    # The /odds response nests teams under the response object
                    # Try to get from fixture-level data
                    row["date"] = fixture_info.get("date", "")[:10]

                if market == "btts":
                    row.update(self._parse_btts_odds(bookmakers))
                else:
                    row.update(
                        self._parse_over_under_odds(bookmakers, market)
                    )

            row["niche_n_bookmakers"] = n_bookmakers
            row["niche_fetch_date"] = datetime.now().strftime("%Y-%m-%d")
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(
            f"{league}: fetched niche odds for {len(df)} fixtures, "
            f"{len(df.columns)} columns"
        )
        return df

    def fetch_and_save(
        self, league: str, fixture_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Fetch niche odds and save to cache.

        Args:
            league: League identifier.
            fixture_ids: Optional list of fixture IDs. If None, fetches upcoming.

        Returns:
            DataFrame with niche odds.
        """
        if fixture_ids is None:
            fixture_ids = self.get_upcoming_fixture_ids(league)

        if not fixture_ids:
            logger.info(f"{league}: no fixtures to fetch odds for")
            return pd.DataFrame()

        df = self.fetch_niche_odds(league, fixture_ids)

        if not df.empty:
            output_path = self.cache_dir / f"{league}_niche_odds.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved niche odds to {output_path}")

        return df
