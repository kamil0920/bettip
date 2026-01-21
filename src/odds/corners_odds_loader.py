"""
Corners Totals Odds Loader

Uses The Odds API for real-time corners over/under odds.
Falls back to estimated odds for historical matches.

Setup:
1. Sign up at https://the-odds-api.com/ (free tier: 500 req/month)
2. Set THE_ODDS_API_KEY in .env

Market typical odds (for 9.5 line):
- Over 9.5 Corners: 1.80-2.00
- Under 9.5 Corners: 1.80-2.00
"""
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "premier_league": "soccer_epl",
    "la_liga": "soccer_spain_la_liga",
    "serie_a": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue_1": "soccer_france_ligue_one",
}

# Common corner lines offered by bookmakers
COMMON_CORNER_LINES = [8.5, 9.5, 10.5, 11.5, 12.5]
DEFAULT_CORNER_LINE = 9.5

# Default odds when real odds unavailable
DEFAULT_OVER_ODDS = 1.90
DEFAULT_UNDER_ODDS = 1.90

ODDS_SOURCE_REAL = "real"
ODDS_SOURCE_ESTIMATED = "estimated"


class CornersOddsLoader:
    """
    Load corners totals odds from The Odds API or use estimated odds.

    Usage:
        loader = CornersOddsLoader()

        # Get current odds for upcoming matches
        odds = loader.get_current_odds("premier_league")

        # Get odds for specific line
        odds = loader.get_current_odds("premier_league", target_line=10.5)

        # Get estimated odds for historical matches
        historical = loader.estimate_historical_odds(matches_df)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_requests: int = 19000,
        warn_at_remaining: int = 1000
    ):
        """
        Initialize loader.

        Args:
            api_key: The Odds API key (or set THE_ODDS_API_KEY env var)
            cache_dir: Directory to cache fetched odds
            max_requests: Maximum requests before stopping (API protection)
            warn_at_remaining: Warn when remaining requests drops below this
        """
        self.api_key = api_key or THE_ODDS_API_KEY
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/corners_odds_cache")
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
                logger.warning(f"Low API quota: only {self.requests_remaining} requests remaining")

        if self.requests_made >= self.max_requests:
            logger.error(f"Reached max_requests limit ({self.max_requests}). Stopping.")
            return False

        return True

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
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

        logger.info(f"API requests: {used} used, {remaining} remaining (session: {self.requests_made})")

        return response.json()

    def get_current_odds(
        self,
        league: str,
        target_line: float = DEFAULT_CORNER_LINE,
        regions: str = "uk,eu",
        bookmakers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get current corners totals odds for upcoming matches.

        Args:
            league: League name (e.g., "premier_league")
            target_line: Corner line to target (e.g., 9.5, 10.5)
            regions: Bookmaker regions (uk, us, eu, au)
            bookmakers: Specific bookmakers to include (optional)

        Returns:
            DataFrame with match info and corners odds
        """
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            raise ValueError(f"Unknown league: {league}. Available: {list(SPORT_KEYS.keys())}")

        # First get list of events
        try:
            events = self._make_request(f"/sports/{sport_key}/events", {})
        except requests.HTTPError as e:
            logger.error(f"Failed to fetch events: {e}")
            return pd.DataFrame()

        if not events:
            logger.info(f"No upcoming events for {league}")
            return pd.DataFrame()

        logger.info(f"Found {len(events)} upcoming events for {league}")

        matches = []
        for event in events:
            event_id = event.get("id")
            params = {
                "regions": regions,
                "markets": "alternate_totals_corners",
                "oddsFormat": "decimal"
            }

            if bookmakers:
                params["bookmakers"] = ",".join(bookmakers)

            try:
                event_data = self._make_request(
                    f"/sports/{sport_key}/events/{event_id}/odds",
                    params
                )
            except requests.HTTPError as e:
                logger.warning(f"Failed to fetch odds for event {event_id}: {e}")
                continue

            match = {
                "event_id": event_id,
                "sport": sport_key,
                "league": league,
                "commence_time": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
            }

            # Parse corners odds for all available lines
            lines_data = self._parse_corners_odds(event_data, target_line)

            if lines_data:
                match.update(lines_data)
                match["odds_source"] = ODDS_SOURCE_REAL
                matches.append(match)

        df = pd.DataFrame(matches)

        if not df.empty:
            cache_file = self.cache_dir / f"{league}_corners_current.csv"
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached {len(df)} matches with corners odds to {cache_file}")

        return df

    def _parse_corners_odds(
        self,
        event_data: Dict,
        target_line: float
    ) -> Optional[Dict]:
        """Parse corners odds from event data, finding closest line to target."""
        all_lines = {}

        for bookmaker in event_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == "alternate_totals_corners":
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
            "corners_line": closest_line,
            "available_lines": available_lines,
        }

        # Add odds for closest line
        line_odds = all_lines[closest_line]
        if line_odds["over"]:
            result["corners_over_avg"] = np.mean(line_odds["over"])
            result["corners_over_max"] = max(line_odds["over"])
            result["corners_over_min"] = min(line_odds["over"])
        if line_odds["under"]:
            result["corners_under_avg"] = np.mean(line_odds["under"])
            result["corners_under_max"] = max(line_odds["under"])
            result["corners_under_min"] = min(line_odds["under"])

        # Also add all lines data for analysis
        for line in COMMON_CORNER_LINES:
            if line in all_lines:
                lo = all_lines[line]
                if lo["over"]:
                    result[f"corners_over_{str(line).replace('.', '_')}_avg"] = np.mean(lo["over"])
                if lo["under"]:
                    result[f"corners_under_{str(line).replace('.', '_')}_avg"] = np.mean(lo["under"])

        return result

    def get_historical_odds(
        self,
        league: str,
        date: str,
        target_line: float = DEFAULT_CORNER_LINE,
        regions: str = "uk,eu"
    ) -> pd.DataFrame:
        """
        Get historical corners odds for a specific date.

        Args:
            league: League name
            date: Date in YYYY-MM-DD format
            target_line: Corner line to target
            regions: Bookmaker regions

        Returns:
            DataFrame with historical corners odds
        """
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            raise ValueError(f"Unknown league: {league}")

        try:
            events_response = self._make_request(
                f"/historical/sports/{sport_key}/events",
                {"date": f"{date}T12:00:00Z"}
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
            try:
                odds_response = self._make_request(
                    f"/historical/sports/{sport_key}/events/{event_id}/odds",
                    {
                        "regions": regions,
                        "markets": "alternate_totals_corners",
                        "oddsFormat": "decimal",
                        "date": f"{date}T12:00:00Z"
                    }
                )
            except requests.HTTPError:
                continue

            event_data = odds_response.get("data", {})
            match = {
                "event_id": event_id,
                "date": date,
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "commence_time": event.get("commence_time"),
            }

            lines_data = self._parse_corners_odds(event_data, target_line)
            if lines_data:
                match.update(lines_data)
                match["odds_source"] = ODDS_SOURCE_REAL
                matches.append(match)

        return pd.DataFrame(matches)

    def estimate_historical_odds(
        self,
        matches_df: pd.DataFrame,
        total_corners_col: str = "total_corners",
        target_line: float = DEFAULT_CORNER_LINE,
        use_market_efficiency: bool = True
    ) -> pd.DataFrame:
        """
        Estimate corners odds for historical matches based on market patterns.

        Args:
            matches_df: DataFrame with match data including total corners
            total_corners_col: Column name for total corners outcome
            target_line: Corner line for odds estimation
            use_market_efficiency: If True, adjust odds based on actual over rate

        Returns:
            DataFrame with estimated corners odds
        """
        df = matches_df.copy()

        base_over = DEFAULT_OVER_ODDS
        base_under = DEFAULT_UNDER_ODDS

        if use_market_efficiency and total_corners_col in df.columns:
            over_rate = (df[total_corners_col] > target_line).mean()

            # Calculate implied probabilities with typical margin
            implied_over = 1 / base_over
            implied_under = 1 / base_under
            total_implied = implied_over + implied_under
            margin = total_implied - 1

            # Adjust based on actual rate
            true_over = over_rate
            true_under = 1 - over_rate

            if true_over > 0 and true_under > 0:
                adj_over = 1 / (true_over + margin * true_over / (true_over + true_under))
                adj_under = 1 / (true_under + margin * true_under / (true_over + true_under))

                logger.info(f"Corners over {target_line} rate: {over_rate:.1%}")
                logger.info(f"Estimated odds: Over={adj_over:.2f}, Under={adj_under:.2f}")

                df["corners_over_odds"] = adj_over
                df["corners_under_odds"] = adj_under
            else:
                df["corners_over_odds"] = base_over
                df["corners_under_odds"] = base_under
        else:
            df["corners_over_odds"] = base_over
            df["corners_under_odds"] = base_under

        # Add variation based on team attacking strength if available
        if "home_corners_won_ema" in df.columns and "away_corners_won_ema" in df.columns:
            # Teams with higher corner EMAs -> higher over odds implied prob
            corner_factor = (df["home_corners_won_ema"] + df["away_corners_won_ema"]) / 10
            corner_factor = corner_factor.clip(0.8, 1.2)

            df["corners_over_odds"] = df["corners_over_odds"] / corner_factor
            df["corners_under_odds"] = df["corners_under_odds"] * corner_factor

        df["corners_line"] = target_line
        df["odds_source"] = ODDS_SOURCE_ESTIMATED

        return df

    def check_api_status(self) -> Dict:
        """Check API key status and remaining requests."""
        if not self.api_key:
            return {"status": "no_key", "message": "API key not configured"}

        try:
            response = requests.get(
                f"{THE_ODDS_API_BASE}/sports",
                params={"apiKey": self.api_key},
                timeout=10
            )
            response.raise_for_status()

            return {
                "status": "ok",
                "requests_used": response.headers.get("x-requests-used", "?"),
                "requests_remaining": response.headers.get("x-requests-remaining", "?"),
                "sports_available": len(response.json())
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


def add_corners_odds_to_features(
    features_df: pd.DataFrame,
    total_corners_col: str = "total_corners",
    target_line: float = DEFAULT_CORNER_LINE,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Add corners odds to existing features DataFrame.

    Args:
        features_df: Features DataFrame with total corners
        total_corners_col: Column name for total corners
        target_line: Corner line for odds
        output_path: Where to save result (optional)

    Returns:
        Features with corners odds added
    """
    loader = CornersOddsLoader()
    result = loader.estimate_historical_odds(features_df, total_corners_col, target_line)

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = CornersOddsLoader()

    status = loader.check_api_status()
    print(f"API Status: {json.dumps(status, indent=2)}")

    if status.get("status") == "ok":
        print("\nFetching Premier League corners odds...")
        odds = loader.get_current_odds("premier_league", target_line=9.5)
        if not odds.empty:
            print(odds[["home_team", "away_team", "corners_line",
                       "corners_over_avg", "corners_under_avg"]].head(10))
        else:
            print("No upcoming matches with corners odds")
    else:
        print("\nUsing estimated odds (no API key)")
        sample = pd.DataFrame({
            "total_corners": [8, 12, 10, 9, 14],
            "home_corners_won_ema": [5.2, 4.8, 5.5, 4.0, 6.0],
            "away_corners_won_ema": [4.5, 5.0, 4.2, 4.5, 5.5]
        })
        result = loader.estimate_historical_odds(sample, target_line=9.5)
        print(result[["total_corners", "corners_over_odds", "corners_under_odds", "corners_line"]])
