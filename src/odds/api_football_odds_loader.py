"""
API-Football Odds Loader

Fetches ALL market odds from API-Football's /odds endpoint in a single call
per fixture (no `bet` parameter = all bet types returned):
- Bet ID 1:  Match Winner (home/draw/away) → h2h_home_avg, h2h_draw_avg, h2h_away_avg
- Bet ID 5:  Over/Under Goals → totals_over_avg, totals_under_avg, totals_line
- Bet ID 8:  Both Teams Score (BTTS) → btts_yes_avg, btts_no_avg
- Bet ID 45: Corners Over/Under
- Bet ID 80: Cards Over/Under
- Bet ID 87: Total Shots On Goal

Uses the existing FootballAPIClient for rate limiting, auth, and retry handling.

Usage:
    loader = ApiFootballOddsLoader()
    df = loader.fetch_all_odds(fixture_ids=[1379200, 1379201])
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

from src.leagues import LEAGUE_IDS

# API-Football bet type IDs we parse
BET_ID_MATCH_WINNER = 1
BET_ID_OVER_UNDER = 5
BET_ID_BTTS = 8
BET_ID_CORNERS = 45
BET_ID_CARDS = 80
BET_ID_SHOTS = 87


def _parse_line_from_value(value: str) -> Optional[float]:
    """Extract numeric line from value string like 'Over 8.5'."""
    match = re.search(r"([\d.]+)", value)
    if match:
        return float(match.group(1))
    return None


class ApiFootballOddsLoader:
    """Loads odds from API-Football /odds endpoint (all bet types in 1 call)."""

    def __init__(self, cache_dir: str = "data/prematch_odds") -> None:
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

    def _fetch_all_odds_for_fixture(
        self, fixture_id: int
    ) -> List[Dict[str, Any]]:
        """Fetch ALL odds for a single fixture (no bet param = all bet types)."""
        response = self.client._make_request(
            "/odds", {"fixture": fixture_id}
        )
        return response.get("response", [])

    def _parse_match_winner(
        self, bookmakers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse Bet ID 1: Match Winner → home/draw/away odds."""
        home_odds: List[float] = []
        draw_odds: List[float] = []
        away_odds: List[float] = []

        for bm in bookmakers:
            for val in bm.get("values", []):
                odd = float(val["odd"])
                v = val["value"]
                if v == "Home":
                    home_odds.append(odd)
                elif v == "Draw":
                    draw_odds.append(odd)
                elif v == "Away":
                    away_odds.append(odd)

        result: Dict[str, Any] = {}
        if home_odds:
            result["h2h_home_avg"] = np.mean(home_odds)
        if draw_odds:
            result["h2h_draw_avg"] = np.mean(draw_odds)
        if away_odds:
            result["h2h_away_avg"] = np.mean(away_odds)
        return result

    def _parse_over_under_goals(
        self, bookmakers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse Bet ID 5: Over/Under Goals for line 2.5."""
        over_odds: List[float] = []
        under_odds: List[float] = []

        for bm in bookmakers:
            for val in bm.get("values", []):
                odd = float(val["odd"])
                value_str = val["value"]
                line = _parse_line_from_value(value_str)
                # Only parse line 2.5 for totals
                if line is not None and abs(line - 2.5) < 0.01:
                    if value_str.startswith("Over"):
                        over_odds.append(odd)
                    elif value_str.startswith("Under"):
                        under_odds.append(odd)

        result: Dict[str, Any] = {}
        if over_odds:
            result["totals_over_avg"] = np.mean(over_odds)
        if under_odds:
            result["totals_under_avg"] = np.mean(under_odds)
        if over_odds or under_odds:
            result["totals_line"] = 2.5
        return result

    def _parse_btts_odds(
        self, bookmakers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse BTTS (Yes/No) odds from bookmaker data."""
        yes_odds: List[float] = []
        no_odds: List[float] = []

        for bm in bookmakers:
            for val in bm.get("values", []):
                odd = float(val["odd"])
                if val["value"] == "Yes":
                    yes_odds.append(odd)
                elif val["value"] == "No":
                    no_odds.append(odd)

        result: Dict[str, Any] = {}
        if yes_odds:
            result["btts_yes_avg"] = np.mean(yes_odds)
            result["niche_btts_yes_avg"] = np.mean(yes_odds)
            result["niche_btts_yes_max"] = np.max(yes_odds)
        if no_odds:
            result["btts_no_avg"] = np.mean(no_odds)
            result["niche_btts_no_avg"] = np.mean(no_odds)
            result["niche_btts_no_max"] = np.max(no_odds)
        return result

    def _parse_over_under_odds(
        self, bookmakers: List[Dict[str, Any]], prefix: str
    ) -> Dict[str, Any]:
        """Parse Over/Under odds with line extraction (corners, cards, shots)."""
        over_odds: List[float] = []
        under_odds: List[float] = []
        lines: List[float] = []

        for bm in bookmakers:
            for val in bm.get("values", []):
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
            from collections import Counter
            line_counts = Counter(lines)
            result[f"niche_{prefix}_line"] = line_counts.most_common(1)[0][0]
        return result

    def _parse_fixture_odds(
        self, data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse all bet types from a single fixture's odds response."""
        row: Dict[str, Any] = {}

        if not data:
            return row

        # Group bets by ID
        bet_map: Dict[int, List[Dict[str, Any]]] = {}
        for item in data:
            for bm in item.get("bookmakers", []):
                for bet in bm.get("bets", []):
                    bet_id = bet.get("id", 0)
                    if bet_id not in bet_map:
                        bet_map[bet_id] = []
                    # Wrap values as a bookmaker-like dict for reuse
                    bet_map[bet_id].append({"values": bet.get("values", [])})

            # Extract team names from response
            if "home_team" not in row:
                fixture_info = item.get("fixture", {})
                row["date"] = fixture_info.get("date", "")[:10]

        # Parse each bet type
        if BET_ID_MATCH_WINNER in bet_map:
            row.update(self._parse_match_winner(bet_map[BET_ID_MATCH_WINNER]))
        if BET_ID_OVER_UNDER in bet_map:
            row.update(self._parse_over_under_goals(bet_map[BET_ID_OVER_UNDER]))
        if BET_ID_BTTS in bet_map:
            row.update(self._parse_btts_odds(bet_map[BET_ID_BTTS]))
        if BET_ID_CORNERS in bet_map:
            row.update(self._parse_over_under_odds(bet_map[BET_ID_CORNERS], "corners"))
        if BET_ID_CARDS in bet_map:
            row.update(self._parse_over_under_odds(bet_map[BET_ID_CARDS], "cards"))
        if BET_ID_SHOTS in bet_map:
            row.update(self._parse_over_under_odds(bet_map[BET_ID_SHOTS], "shots_ot"))

        return row

    def fetch_all_odds(
        self,
        fixture_ids: List[int],
        team_names: Optional[Dict[int, tuple]] = None,
    ) -> pd.DataFrame:
        """Fetch all odds for a list of fixtures (1 API call per fixture).

        Args:
            fixture_ids: List of API-Football fixture IDs.
            team_names: Optional mapping of fixture_id → (home_team, away_team).

        Returns:
            DataFrame with one row per fixture, columns for all parsed odds.
        """
        rows: List[Dict[str, Any]] = []

        for fid in fixture_ids:
            try:
                data = self._fetch_all_odds_for_fixture(fid)
            except Exception as e:
                logger.warning(f"Failed to fetch odds for fixture {fid}: {e}")
                continue

            row = self._parse_fixture_odds(data)
            row["fixture_id"] = fid

            # Add team names if provided
            if team_names and fid in team_names:
                row["home_team"] = team_names[fid][0]
                row["away_team"] = team_names[fid][1]

            row["fetch_date"] = datetime.now().strftime("%Y-%m-%d")

            if row:
                rows.append(row)
                n_markets = sum(1 for k in row if k.startswith(("h2h_", "totals_", "btts_", "niche_")))
                logger.info(f"  fixture {fid}: {n_markets} odds columns")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(f"Fetched odds for {len(df)} fixtures, {len(df.columns)} columns")
        return df

    def fetch_and_save(
        self,
        fixture_ids: List[int],
        team_names: Optional[Dict[int, tuple]] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch all odds and save to parquet.

        Args:
            fixture_ids: List of fixture IDs.
            team_names: Optional mapping of fixture_id → (home_team, away_team).
            output_path: Custom output path. Defaults to odds_latest.parquet.

        Returns:
            DataFrame with all odds.
        """
        if not fixture_ids:
            logger.info("No fixtures to fetch odds for")
            return pd.DataFrame()

        df = self.fetch_all_odds(fixture_ids, team_names=team_names)

        if not df.empty:
            out = Path(output_path) if output_path else self.cache_dir / "odds_latest.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            logger.info(f"Saved odds to {out}")

        return df

    # Legacy compatibility
    def fetch_niche_odds(
        self, league: str, fixture_ids: List[int]
    ) -> pd.DataFrame:
        """Fetch odds for a list of fixtures (legacy interface)."""
        return self.fetch_all_odds(fixture_ids)
