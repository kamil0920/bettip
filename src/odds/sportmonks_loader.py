"""
SportMonks API integration for niche market odds (corners, cards, shots).

Fetches pre-match odds from SportMonks API for specialty betting markets
that are not available from football-data.co.uk.

API Documentation: https://docs.sportmonks.com/football
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# SportMonks league IDs for supported leagues
SPORTMONKS_LEAGUES = {
    "premier_league": 8,
    "bundesliga": 82,
    "ligue_1": 301,
    "serie_a": 384,
    "la_liga": 564,
}

# Market IDs for niche betting markets
NICHE_MARKET_IDS = {
    # Corners markets
    "corners_over_under": 67,
    "corners_asian_total": 61,
    "corners_asian_handicap": 62,
    "corners_alternative": 69,
    "corners_1x2": 269,
    "corners_match_bet": 71,
    "corners_handicap": 72,
    "corners_first_half": 70,
    "corners_team": 74,
    # Cards markets
    "cards_over_under": 255,
    "cards_asian_total": 272,
    "cards_asian_handicap": 273,
    "cards_handicap": 277,
    # Shots markets (match level)
    "match_shots_on_target": 291,
    "match_shots": 292,
    "team_shots_on_target": 284,
    "team_shots": 285,
}

# Primary markets for each niche type
PRIMARY_MARKETS = {
    "corners": [67, 69],  # Over/Under and Alternative
    "cards": [255],       # Over/Under
    "shots": [291, 292, 284, 285],  # Match and team shots
}


@dataclass
class OddsEntry:
    """Single odds entry from SportMonks."""
    fixture_id: int
    market_id: int
    market_name: str
    label: str
    line: Optional[float]
    odds: float
    bookmaker_id: int
    bookmaker_name: Optional[str] = None


class SportMonksLoader:
    """
    Load niche market odds from SportMonks API.

    Usage:
        loader = SportMonksLoader()

        # Get upcoming fixtures with corners odds
        df = loader.get_upcoming_corners_odds("premier_league", days_ahead=7)

        # Get odds for specific fixture
        odds = loader.get_fixture_niche_odds(fixture_id=19427671)
    """

    BASE_URL = "https://api.sportmonks.com/v3"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize SportMonks loader.

        Args:
            api_key: SportMonks API key. If None, reads from SPORTSMONK_KEY env var.
        """
        self.api_key = api_key or os.getenv("SPORTSMONK_KEY")
        if not self.api_key:
            raise ValueError(
                "SportMonks API key required. Set SPORTSMONK_KEY environment variable "
                "or pass api_key parameter."
            )
        self._bookmaker_cache: Dict[int, str] = {}

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Make authenticated request to SportMonks API."""
        url = f"{self.BASE_URL}/{endpoint}"

        request_params = {"api_token": self.api_key}
        if params:
            request_params.update(params)

        logger.debug(f"Requesting: {url}")

        response = requests.get(url, params=request_params, timeout=timeout)
        response.raise_for_status()

        return response.json()

    def get_leagues(self) -> List[Dict[str, Any]]:
        """Get list of available leagues."""
        data = self._request("football/leagues")
        return data.get("data", [])

    def get_subscription_info(self) -> Dict[str, Any]:
        """Get current subscription details."""
        data = self._request("football/leagues")
        return data.get("subscription", {})

    def get_fixtures_between(
        self,
        start_date: datetime,
        end_date: datetime,
        league_ids: Optional[List[int]] = None,
        include_odds: bool = True,
        per_page: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get fixtures between two dates.

        Args:
            start_date: Start date
            end_date: End date
            league_ids: Filter by league IDs (optional)
            include_odds: Include odds data
            per_page: Results per page

        Returns:
            List of fixture dictionaries
        """
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        params = {"per_page": per_page}

        if include_odds:
            params["include"] = "odds"

        if league_ids:
            params["filters"] = f"fixtureLeagues:{','.join(map(str, league_ids))}"

        endpoint = f"football/fixtures/between/{start_str}/{end_str}"

        all_fixtures = []
        page = 1

        while True:
            params["page"] = page
            data = self._request(endpoint, params)

            fixtures = data.get("data", [])
            if not fixtures:
                break

            all_fixtures.extend(fixtures)

            # Check pagination
            pagination = data.get("pagination", {})
            if page >= pagination.get("last_page", 1):
                break

            page += 1

        logger.info(f"Fetched {len(all_fixtures)} fixtures between {start_str} and {end_str}")
        return all_fixtures

    def get_fixture_odds(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Get all odds for a specific fixture."""
        data = self._request(
            f"football/fixtures/{fixture_id}",
            params={"include": "odds"}
        )
        return data.get("data", {}).get("odds", [])

    def _extract_niche_odds(
        self,
        odds_list: List[Dict[str, Any]],
        market_ids: List[int]
    ) -> List[OddsEntry]:
        """Extract odds entries for specific market IDs."""
        entries = []

        for odds in odds_list:
            market_id = odds.get("market_id")
            if market_id not in market_ids:
                continue

            line = odds.get("total") or odds.get("handicap")
            if line is not None:
                try:
                    line = float(line)
                except (ValueError, TypeError):
                    line = None

            value = odds.get("value")
            if value is None:
                continue

            try:
                value = float(value)
            except (ValueError, TypeError):
                continue

            entry = OddsEntry(
                fixture_id=odds.get("fixture_id", 0),
                market_id=market_id,
                market_name=odds.get("market_description", "Unknown"),
                label=odds.get("label", ""),
                line=line,
                odds=value,
                bookmaker_id=odds.get("bookmaker_id", 0),
            )
            entries.append(entry)

        return entries

    def get_corners_odds(
        self,
        fixture_id: Optional[int] = None,
        fixtures: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Get corners betting odds.

        Args:
            fixture_id: Single fixture ID
            fixtures: List of fixture dicts with odds included

        Returns:
            DataFrame with corners odds by line
        """
        if fixture_id and not fixtures:
            odds_list = self.get_fixture_odds(fixture_id)
            fixtures = [{"id": fixture_id, "odds": odds_list}]

        if not fixtures:
            return pd.DataFrame()

        rows = []

        for fixture in fixtures:
            fix_id = fixture.get("id")
            fix_name = fixture.get("name", "")
            start_time = fixture.get("starting_at", "")
            odds_list = fixture.get("odds", [])

            if not odds_list:
                continue

            entries = self._extract_niche_odds(odds_list, PRIMARY_MARKETS["corners"])

            # Group by line
            lines_data: Dict[float, Dict[str, List[float]]] = {}

            for entry in entries:
                if entry.line is None:
                    continue

                if entry.line not in lines_data:
                    lines_data[entry.line] = {"over": [], "under": []}

                label_lower = entry.label.lower()
                if "over" in label_lower:
                    lines_data[entry.line]["over"].append(entry.odds)
                elif "under" in label_lower:
                    lines_data[entry.line]["under"].append(entry.odds)

            # Create rows for each line
            for line, odds_dict in sorted(lines_data.items()):
                over_odds = odds_dict["over"]
                under_odds = odds_dict["under"]

                row = {
                    "fixture_id": fix_id,
                    "fixture_name": fix_name,
                    "start_time": start_time,
                    "market": "corners",
                    "line": line,
                    "over_best": min(over_odds) if over_odds else None,
                    "over_worst": max(over_odds) if over_odds else None,
                    "over_avg": sum(over_odds) / len(over_odds) if over_odds else None,
                    "over_count": len(over_odds),
                    "under_best": max(under_odds) if under_odds else None,
                    "under_worst": min(under_odds) if under_odds else None,
                    "under_avg": sum(under_odds) / len(under_odds) if under_odds else None,
                    "under_count": len(under_odds),
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def get_cards_odds(
        self,
        fixture_id: Optional[int] = None,
        fixtures: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Get cards betting odds.

        Args:
            fixture_id: Single fixture ID
            fixtures: List of fixture dicts with odds included

        Returns:
            DataFrame with cards odds by line
        """
        if fixture_id and not fixtures:
            odds_list = self.get_fixture_odds(fixture_id)
            fixtures = [{"id": fixture_id, "odds": odds_list}]

        if not fixtures:
            return pd.DataFrame()

        rows = []

        for fixture in fixtures:
            fix_id = fixture.get("id")
            fix_name = fixture.get("name", "")
            start_time = fixture.get("starting_at", "")
            odds_list = fixture.get("odds", [])

            if not odds_list:
                continue

            entries = self._extract_niche_odds(odds_list, PRIMARY_MARKETS["cards"])

            # Group by line
            lines_data: Dict[float, Dict[str, List[float]]] = {}

            for entry in entries:
                if entry.line is None:
                    continue

                if entry.line not in lines_data:
                    lines_data[entry.line] = {"over": [], "under": []}

                label_lower = entry.label.lower()
                if "over" in label_lower:
                    lines_data[entry.line]["over"].append(entry.odds)
                elif "under" in label_lower:
                    lines_data[entry.line]["under"].append(entry.odds)

            # Create rows for each line
            for line, odds_dict in sorted(lines_data.items()):
                over_odds = odds_dict["over"]
                under_odds = odds_dict["under"]

                row = {
                    "fixture_id": fix_id,
                    "fixture_name": fix_name,
                    "start_time": start_time,
                    "market": "cards",
                    "line": line,
                    "over_best": min(over_odds) if over_odds else None,
                    "over_worst": max(over_odds) if over_odds else None,
                    "over_avg": sum(over_odds) / len(over_odds) if over_odds else None,
                    "over_count": len(over_odds),
                    "under_best": max(under_odds) if under_odds else None,
                    "under_worst": min(under_odds) if under_odds else None,
                    "under_avg": sum(under_odds) / len(under_odds) if under_odds else None,
                    "under_count": len(under_odds),
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def get_upcoming_niche_odds(
        self,
        leagues: Optional[List[str]] = None,
        days_ahead: int = 7,
        markets: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get upcoming fixtures with niche market odds.

        Args:
            leagues: List of league names (e.g., ["premier_league", "la_liga"])
                    If None, uses all supported leagues.
            days_ahead: Number of days to look ahead
            markets: List of markets to include ("corners", "cards", "shots")
                    If None, includes all.

        Returns:
            DataFrame with niche odds for upcoming fixtures
        """
        if leagues is None:
            leagues = list(SPORTMONKS_LEAGUES.keys())

        if markets is None:
            markets = ["corners", "cards"]

        league_ids = [SPORTMONKS_LEAGUES[lg] for lg in leagues if lg in SPORTMONKS_LEAGUES]

        if not league_ids:
            logger.warning(f"No valid leagues found in: {leagues}")
            return pd.DataFrame()

        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_ahead)

        fixtures = self.get_fixtures_between(
            start_date=start_date,
            end_date=end_date,
            league_ids=league_ids,
            include_odds=True
        )

        if not fixtures:
            logger.warning("No fixtures found")
            return pd.DataFrame()

        dfs = []

        if "corners" in markets:
            corners_df = self.get_corners_odds(fixtures=fixtures)
            if not corners_df.empty:
                dfs.append(corners_df)

        if "cards" in markets:
            cards_df = self.get_cards_odds(fixtures=fixtures)
            if not cards_df.empty:
                dfs.append(cards_df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Retrieved {len(combined)} odds entries for {len(fixtures)} fixtures")

        return combined

    def get_best_odds_for_line(
        self,
        fixture_id: int,
        market: str,
        line: float,
        side: str = "over"
    ) -> Optional[float]:
        """
        Get best available odds for a specific line.

        Args:
            fixture_id: Fixture ID
            market: "corners" or "cards"
            line: The line (e.g., 9.5, 10.5)
            side: "over" or "under"

        Returns:
            Best odds value or None
        """
        if market == "corners":
            df = self.get_corners_odds(fixture_id=fixture_id)
        elif market == "cards":
            df = self.get_cards_odds(fixture_id=fixture_id)
        else:
            raise ValueError(f"Unknown market: {market}")

        if df.empty:
            return None

        line_df = df[df["line"] == line]
        if line_df.empty:
            return None

        col = f"{side}_best"
        if col in line_df.columns:
            return line_df[col].iloc[0]

        return None

    def format_for_recommendations(
        self,
        fixture_id: int,
        fixture_name: str,
        league: str,
        market: str,
        line: float,
        side: str,
        our_probability: float
    ) -> Optional[Dict[str, Any]]:
        """
        Format odds data for betting recommendations.

        Args:
            fixture_id: SportMonks fixture ID
            fixture_name: Match name
            league: League name
            market: "corners" or "cards"
            line: Betting line
            side: "over" or "under"
            our_probability: Model's predicted probability

        Returns:
            Dictionary with recommendation data including edge calculation
        """
        best_odds = self.get_best_odds_for_line(fixture_id, market, line, side)

        if best_odds is None:
            return None

        # Calculate implied probability and edge
        implied_prob = 1 / best_odds
        edge = our_probability - implied_prob
        edge_pct = edge * 100

        return {
            "fixture_id": fixture_id,
            "fixture_name": fixture_name,
            "league": league,
            "market": market.upper(),
            "side": side.upper(),
            "line": line,
            "best_odds": best_odds,
            "implied_probability": implied_prob,
            "our_probability": our_probability,
            "edge": edge,
            "edge_pct": edge_pct,
            "is_value_bet": edge > 0.05,  # 5% minimum edge
        }


# Team name mapping between SportMonks and existing data
SPORTMONKS_TEAM_MAPPING = {
    # Premier League
    "Manchester United": "Manchester United",
    "Manchester City": "Manchester City",
    "Liverpool": "Liverpool",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle United",
    "West Ham United": "West Ham United",
    "Aston Villa": "Aston Villa",
    "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nottingham Forest",
    "Fulham": "Fulham",
    "Brentford": "Brentford",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Bournemouth": "Bournemouth",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Southampton": "Southampton",
    # La Liga
    "Real Madrid": "Real Madrid",
    "FC Barcelona": "Barcelona",
    "Atletico Madrid": "Atletico Madrid",
    "Sevilla FC": "Sevilla",
    "Real Betis": "Real Betis",
    "Real Sociedad": "Real Sociedad",
    "Villarreal CF": "Villarreal",
    "Athletic Club": "Athletic Club",
    # Serie A
    "Inter": "Inter",
    "AC Milan": "AC Milan",
    "Juventus": "Juventus",
    "SSC Napoli": "Napoli",
    "AS Roma": "AS Roma",
    "SS Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "ACF Fiorentina": "Fiorentina",
    # Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "VfL Wolfsburg": "Wolfsburg",
    "Borussia Mönchengladbach": "Borussia Monchengladbach",
    # Ligue 1
    "Paris Saint-Germain": "Paris Saint Germain",
    "Olympique Marseille": "Marseille",
    "Olympique Lyon": "Lyon",
    "AS Monaco": "Monaco",
    "LOSC Lille": "Lille",
}


def normalize_sportmonks_team(name: str) -> str:
    """Normalize SportMonks team name to match existing data."""
    return SPORTMONKS_TEAM_MAPPING.get(name, name)
