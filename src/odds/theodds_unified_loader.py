"""
The Odds API Unified Loader

Fetches odds for multiple markets from The Odds API in a single efficient workflow:
- BTTS (Both Teams To Score)
- Corners (alternate_totals_corners)
- Cards (player_cards)
- Shots (player_shots_on_target)

Setup:
1. Sign up at https://the-odds-api.com/ ($25/month for 20K requests)
2. Set THE_ODDS_API_KEY in .env

Usage:
    loader = TheOddsUnifiedLoader()

    # Get odds for all markets at once
    odds_df = loader.fetch_all_markets("premier_league")

    # Get specific market
    btts_df = loader.fetch_market("premier_league", "btts")
    corners_df = loader.fetch_market("premier_league", "alternate_totals_corners")
"""
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# League to The Odds API sport key mapping
SPORT_KEYS = {
    "premier_league": "soccer_epl",
    "la_liga": "soccer_spain_la_liga",
    "serie_a": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue_1": "soccer_france_ligue_one",
    "ekstraklasa": "soccer_poland_ekstraklasa",
}

# Market keys supported by The Odds API
MARKET_KEYS = {
    "btts": "btts",
    "corners": "alternate_totals_corners",
    "cards": "alternate_totals_cards",
    "shots": "player_shots_on_target",
}

# Default lines for totals markets
DEFAULT_LINES = {
    "alternate_totals_corners": 9.5,
    "player_cards": 4.5,
    "player_shots_on_target": 4.5,
}

# Odds source constants
ODDS_SOURCE_REAL = "real"
ODDS_SOURCE_ESTIMATED = "estimated"


class TheOddsUnifiedLoader:
    """
    Unified loader for fetching multiple betting markets from The Odds API.

    Provides efficient multi-market fetching with:
    - Single API call for multiple markets where possible
    - Request rate limiting and quota tracking
    - Caching to Parquet format
    - Fallback to estimated odds when API unavailable
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_requests: int = 19000,
        warn_at_remaining: int = 1000,
    ):
        """
        Initialize the unified loader.

        Args:
            api_key: The Odds API key (or set THE_ODDS_API_KEY env var)
            cache_dir: Directory to cache fetched odds
            max_requests: Maximum requests before stopping (API protection)
            warn_at_remaining: Warn when remaining requests drops below this
        """
        self.api_key = api_key or THE_ODDS_API_KEY
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/theodds_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_requests = max_requests
        self.warn_at_remaining = warn_at_remaining
        self.requests_made = 0
        self.requests_remaining = None

        if not self.api_key:
            logger.warning(
                "No API key set. Only estimated odds will be available. "
                "Set THE_ODDS_API_KEY in .env or pass api_key parameter."
            )

    def _check_rate_limit(self) -> bool:
        """Check if we should stop making requests."""
        if self.requests_remaining is not None:
            if self.requests_remaining <= 0:
                logger.error("API quota exhausted! No requests remaining.")
                return False
            if self.requests_remaining < self.warn_at_remaining:
                logger.warning(
                    f"Low API quota: only {self.requests_remaining} requests remaining"
                )

        if self.requests_made >= self.max_requests:
            logger.error(
                f"Reached max_requests limit ({self.max_requests}). Stopping."
            )
            return False

        return True

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Any]:
        """Make API request to The Odds API with rate limiting."""
        if not self.api_key:
            raise ValueError("API key not set")

        if not self._check_rate_limit():
            raise RuntimeError(
                f"API rate limit protection triggered. "
                f"Made {self.requests_made} requests, {self.requests_remaining} remaining."
            )

        params["apiKey"] = self.api_key
        url = f"{THE_ODDS_API_BASE}{endpoint}"

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        self.requests_made += 1
        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")

        if remaining is not None:
            self.requests_remaining = int(remaining)

        logger.info(
            f"API requests: {used} used, {remaining} remaining "
            f"(session: {self.requests_made})"
        )

        return response.json()

    def fetch_all_markets(
        self,
        league: str,
        regions: str = "uk,eu",
        save_to_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch odds for all supported markets for a league.

        Args:
            league: League name (e.g., "premier_league")
            regions: Bookmaker regions to include
            save_to_cache: Whether to save results to cache

        Returns:
            DataFrame with all market odds merged by event
        """
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            raise ValueError(
                f"Unknown league: {league}. Available: {list(SPORT_KEYS.keys())}"
            )

        # Fetch events first
        try:
            events = self._make_request(f"/sports/{sport_key}/events", {})
        except requests.HTTPError as e:
            logger.error(f"Failed to fetch events: {e}")
            return pd.DataFrame()

        if not events:
            logger.info(f"No upcoming events for {league}")
            return pd.DataFrame()

        logger.info(f"Found {len(events)} upcoming events for {league}")

        # Fetch all markets for each event
        all_matches = []
        for event in events:
            event_id = event.get("id")
            match_data = {
                "event_id": event_id,
                "sport": sport_key,
                "league": league,
                "commence_time": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
            }

            # Fetch BTTS
            btts_odds = self._fetch_btts_odds(sport_key, event_id, regions)
            if btts_odds:
                match_data.update(btts_odds)

            # Fetch Corners
            corners_odds = self._fetch_totals_odds(
                sport_key, event_id, "alternate_totals_corners", regions
            )
            if corners_odds:
                match_data.update({f"corners_{k}": v for k, v in corners_odds.items()})

            # Fetch Cards
            cards_odds = self._fetch_totals_odds(
                sport_key, event_id, "player_cards", regions
            )
            if cards_odds:
                match_data.update({f"cards_{k}": v for k, v in cards_odds.items()})

            # Fetch Shots
            shots_odds = self._fetch_totals_odds(
                sport_key, event_id, "player_shots_on_target", regions
            )
            if shots_odds:
                match_data.update({f"shots_{k}": v for k, v in shots_odds.items()})

            # Only add if we got at least one market
            if any(
                k in match_data
                for k in ["btts_yes_avg", "corners_over_avg", "cards_over_avg", "shots_over_avg"]
            ):
                match_data["odds_source"] = ODDS_SOURCE_REAL
                match_data["fetch_timestamp"] = datetime.utcnow().isoformat()
                all_matches.append(match_data)

        df = pd.DataFrame(all_matches)

        if not df.empty and save_to_cache:
            cache_file = self.cache_dir / f"{league}_all_markets.parquet"
            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached {len(df)} matches to {cache_file}")

            # Also save CSV for debugging
            csv_file = self.cache_dir / f"{league}_all_markets.csv"
            df.to_csv(csv_file, index=False)

        return df

    def _fetch_btts_odds(
        self, sport_key: str, event_id: str, regions: str
    ) -> Optional[Dict]:
        """Fetch BTTS odds for a single event."""
        try:
            event_data = self._make_request(
                f"/sports/{sport_key}/events/{event_id}/odds",
                {"regions": regions, "markets": "btts", "oddsFormat": "decimal"},
            )
        except requests.HTTPError as e:
            logger.warning(f"Failed to fetch BTTS odds for {event_id}: {e}")
            return None

        btts_yes_odds = []
        btts_no_odds = []

        for bookmaker in event_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == "btts":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Yes":
                            btts_yes_odds.append(outcome.get("price"))
                        elif outcome.get("name") == "No":
                            btts_no_odds.append(outcome.get("price"))

        if not btts_yes_odds and not btts_no_odds:
            return None

        result = {}
        if btts_yes_odds:
            result["btts_yes_avg"] = np.mean(btts_yes_odds)
            result["btts_yes_max"] = max(btts_yes_odds)
            result["btts_yes_min"] = min(btts_yes_odds)
        if btts_no_odds:
            result["btts_no_avg"] = np.mean(btts_no_odds)
            result["btts_no_max"] = max(btts_no_odds)
            result["btts_no_min"] = min(btts_no_odds)

        return result

    def _fetch_totals_odds(
        self,
        sport_key: str,
        event_id: str,
        market_key: str,
        regions: str,
        target_line: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Fetch totals odds (corners, cards, shots) for a single event.

        Args:
            sport_key: The Odds API sport key
            event_id: Event ID
            market_key: Market key (alternate_totals_corners, player_cards, etc.)
            regions: Bookmaker regions
            target_line: Target line (uses DEFAULT_LINES if not specified)

        Returns:
            Dict with over/under odds for the market
        """
        if target_line is None:
            target_line = DEFAULT_LINES.get(market_key, 9.5)

        try:
            event_data = self._make_request(
                f"/sports/{sport_key}/events/{event_id}/odds",
                {"regions": regions, "markets": market_key, "oddsFormat": "decimal"},
            )
        except requests.HTTPError as e:
            logger.debug(f"Failed to fetch {market_key} odds for {event_id}: {e}")
            return None

        all_lines = {}

        for bookmaker in event_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == market_key:
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price")
                        point = outcome.get("point")

                        if point is not None and price is not None:
                            if point not in all_lines:
                                all_lines[point] = {"over": [], "under": []}

                            if "Over" in name:
                                all_lines[point]["over"].append(price)
                            elif "Under" in name:
                                all_lines[point]["under"].append(price)

        if not all_lines:
            return None

        # Find closest available line to target
        available_lines = sorted(all_lines.keys())
        closest_line = min(available_lines, key=lambda x: abs(x - target_line))

        result = {
            "line": closest_line,
            "available_lines": available_lines,
        }

        line_odds = all_lines[closest_line]
        if line_odds["over"]:
            result["over_avg"] = np.mean(line_odds["over"])
            result["over_max"] = max(line_odds["over"])
            result["over_min"] = min(line_odds["over"])
        if line_odds["under"]:
            result["under_avg"] = np.mean(line_odds["under"])
            result["under_max"] = max(line_odds["under"])
            result["under_min"] = min(line_odds["under"])

        return result

    def fetch_market(
        self,
        league: str,
        market: str,
        regions: str = "uk,eu",
        target_line: Optional[float] = None,
        save_to_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch odds for a specific market.

        Args:
            league: League name
            market: Market name (btts, corners, cards, shots)
            regions: Bookmaker regions
            target_line: Target line for totals markets
            save_to_cache: Whether to save to cache

        Returns:
            DataFrame with market odds
        """
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            raise ValueError(f"Unknown league: {league}")

        # Map market name to key
        market_key = MARKET_KEYS.get(market, market)

        try:
            events = self._make_request(f"/sports/{sport_key}/events", {})
        except requests.HTTPError as e:
            logger.error(f"Failed to fetch events: {e}")
            return pd.DataFrame()

        if not events:
            return pd.DataFrame()

        matches = []
        for event in events:
            event_id = event.get("id")
            match_data = {
                "event_id": event_id,
                "sport": sport_key,
                "league": league,
                "commence_time": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
            }

            if market_key == "btts":
                odds = self._fetch_btts_odds(sport_key, event_id, regions)
            else:
                odds = self._fetch_totals_odds(
                    sport_key, event_id, market_key, regions, target_line
                )

            if odds:
                match_data.update(odds)
                match_data["odds_source"] = ODDS_SOURCE_REAL
                match_data["fetch_timestamp"] = datetime.utcnow().isoformat()
                matches.append(match_data)

        df = pd.DataFrame(matches)

        if not df.empty and save_to_cache:
            cache_file = self.cache_dir / f"{league}_{market}.parquet"
            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached {len(df)} {market} odds to {cache_file}")

        return df

    def fetch_historical_odds(
        self,
        league: str,
        date: str,
        markets: Optional[List[str]] = None,
        regions: str = "uk,eu",
    ) -> pd.DataFrame:
        """
        Fetch historical odds for a specific date.

        Args:
            league: League name
            date: Date in YYYY-MM-DD format
            markets: List of markets to fetch (default: all)
            regions: Bookmaker regions

        Returns:
            DataFrame with historical odds
        """
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            raise ValueError(f"Unknown league: {league}")

        if markets is None:
            markets = list(MARKET_KEYS.keys())

        try:
            events_response = self._make_request(
                f"/historical/sports/{sport_key}/events",
                {"date": f"{date}T12:00:00Z"},
            )
        except requests.HTTPError as e:
            logger.warning(f"Failed to fetch historical events: {e}")
            return pd.DataFrame()

        events = events_response.get("data", [])
        if not events:
            return pd.DataFrame()

        matches = []
        for event in events:
            event_id = event.get("id")
            match_data = {
                "event_id": event_id,
                "date": date,
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "commence_time": event.get("commence_time"),
            }

            # Fetch each requested market
            for market in markets:
                market_key = MARKET_KEYS.get(market, market)
                try:
                    odds_response = self._make_request(
                        f"/historical/sports/{sport_key}/events/{event_id}/odds",
                        {
                            "regions": regions,
                            "markets": market_key,
                            "oddsFormat": "decimal",
                            "date": f"{date}T12:00:00Z",
                        },
                    )
                except requests.HTTPError:
                    continue

                event_data = odds_response.get("data", {})
                if market_key == "btts":
                    odds = self._parse_btts_from_response(event_data)
                else:
                    odds = self._parse_totals_from_response(event_data, market_key)

                if odds:
                    prefix = market if market in MARKET_KEYS else ""
                    if prefix:
                        match_data.update({f"{prefix}_{k}": v for k, v in odds.items()})
                    else:
                        match_data.update(odds)

            if len(match_data) > 5:  # Has more than just base fields
                match_data["odds_source"] = ODDS_SOURCE_REAL
                matches.append(match_data)

        return pd.DataFrame(matches)

    def _parse_btts_from_response(self, event_data: Dict) -> Optional[Dict]:
        """Parse BTTS odds from historical API response."""
        btts_yes_odds = []
        btts_no_odds = []

        for bookmaker in event_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == "btts":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Yes":
                            btts_yes_odds.append(outcome.get("price"))
                        elif outcome.get("name") == "No":
                            btts_no_odds.append(outcome.get("price"))

        if not btts_yes_odds and not btts_no_odds:
            return None

        result = {}
        if btts_yes_odds:
            result["yes_avg"] = np.mean(btts_yes_odds)
        if btts_no_odds:
            result["no_avg"] = np.mean(btts_no_odds)

        return result

    def _parse_totals_from_response(
        self, event_data: Dict, market_key: str
    ) -> Optional[Dict]:
        """Parse totals odds from historical API response."""
        target_line = DEFAULT_LINES.get(market_key, 9.5)
        all_lines = {}

        for bookmaker in event_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == market_key:
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price")
                        point = outcome.get("point")

                        if point is not None and price is not None:
                            if point not in all_lines:
                                all_lines[point] = {"over": [], "under": []}

                            if "Over" in name:
                                all_lines[point]["over"].append(price)
                            elif "Under" in name:
                                all_lines[point]["under"].append(price)

        if not all_lines:
            return None

        available_lines = sorted(all_lines.keys())
        closest_line = min(available_lines, key=lambda x: abs(x - target_line))

        result = {"line": closest_line}
        line_odds = all_lines[closest_line]
        if line_odds["over"]:
            result["over_avg"] = np.mean(line_odds["over"])
        if line_odds["under"]:
            result["under_avg"] = np.mean(line_odds["under"])

        return result

    def check_api_status(self) -> Dict:
        """Check API key status and remaining requests."""
        if not self.api_key:
            return {"status": "no_key", "message": "API key not configured"}

        try:
            response = requests.get(
                f"{THE_ODDS_API_BASE}/sports",
                params={"apiKey": self.api_key},
                timeout=10,
            )
            response.raise_for_status()

            return {
                "status": "ok",
                "requests_used": response.headers.get("x-requests-used", "?"),
                "requests_remaining": response.headers.get("x-requests-remaining", "?"),
                "sports_available": len(response.json()),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_available_markets(self, league: str) -> List[str]:
        """
        Get list of markets available for a league.

        Args:
            league: League name

        Returns:
            List of available market keys
        """
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            return []

        try:
            # Check first event for available markets
            events = self._make_request(f"/sports/{sport_key}/events", {})
            if not events:
                return []

            event_id = events[0].get("id")
            available = []

            for market_name, market_key in MARKET_KEYS.items():
                try:
                    response = self._make_request(
                        f"/sports/{sport_key}/events/{event_id}/odds",
                        {"markets": market_key, "oddsFormat": "decimal"},
                    )
                    if response.get("bookmakers"):
                        available.append(market_name)
                except requests.HTTPError:
                    pass

            return available
        except Exception as e:
            logger.error(f"Failed to check available markets: {e}")
            return []

    def load_from_cache(self, league: str, market: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load cached odds from Parquet file.

        Args:
            league: League name
            market: Specific market or None for all markets

        Returns:
            DataFrame with cached odds or None if not found
        """
        if market:
            cache_file = self.cache_dir / f"{league}_{market}.parquet"
        else:
            cache_file = self.cache_dir / f"{league}_all_markets.parquet"

        if cache_file.exists():
            return pd.read_parquet(cache_file)

        return None


def fetch_odds_for_predictions(
    leagues: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch current odds for all leagues and markets.

    Convenience function for use in prediction pipelines.

    Args:
        leagues: Leagues to fetch (default: all)
        output_dir: Directory to save results

    Returns:
        Combined DataFrame with all odds
    """
    if leagues is None:
        leagues = list(SPORT_KEYS.keys())

    loader = TheOddsUnifiedLoader()

    all_odds = []
    for league in leagues:
        logger.info(f"Fetching odds for {league}...")
        df = loader.fetch_all_markets(league)
        if not df.empty:
            all_odds.append(df)

    if not all_odds:
        return pd.DataFrame()

    combined = pd.concat(all_odds, ignore_index=True)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"theodds_all_{datetime.now().strftime('%Y%m%d')}.parquet"
        combined.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(combined)} odds to {output_file}")

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = TheOddsUnifiedLoader()

    status = loader.check_api_status()
    print(f"API Status: {json.dumps(status, indent=2)}")

    if status.get("status") == "ok":
        print("\nFetching Premier League odds for all markets...")
        odds = loader.fetch_all_markets("premier_league")

        if not odds.empty:
            print(f"\nFetched {len(odds)} matches with odds:")
            print(odds[["home_team", "away_team"]].head())

            # Show market coverage
            for market in ["btts", "corners", "cards", "shots"]:
                col = f"{market}_" if market != "btts" else "btts_"
                coverage = odds[[c for c in odds.columns if c.startswith(col)]].notna().any(axis=1).sum()
                print(f"  {market}: {coverage}/{len(odds)} matches")
        else:
            print("No upcoming matches with odds")
    else:
        print("\nAPI not configured. Set THE_ODDS_API_KEY in .env")
