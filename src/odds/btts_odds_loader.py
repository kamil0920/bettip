"""
BTTS (Both Teams To Score) Odds Loader

Uses The Odds API for real-time BTTS odds.
Falls back to estimated odds for historical matches.

Setup:
1. Sign up at https://the-odds-api.com/ (free tier: 500 req/month)
2. Set THE_ODDS_API_KEY in .env

Market typical odds:
- BTTS Yes: 1.65-2.00 (avg ~1.80)
- BTTS No: 1.85-2.20 (avg ~2.00)
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

DEFAULT_BTTS_YES_ODDS = 1.80
DEFAULT_BTTS_NO_ODDS = 2.00

# Odds source constants
ODDS_SOURCE_REAL = "real"
ODDS_SOURCE_ESTIMATED = "estimated"


class BTTSOddsLoader:
    """
    Load BTTS odds from The Odds API or use estimated odds.

    Usage:
        loader = BTTSOddsLoader()

        # Get current odds for upcoming matches
        odds = loader.get_current_odds("premier_league")

        # Get estimated odds for historical matches
        historical = loader.estimate_historical_odds(matches_df)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_requests: int = 19000,  # Safety buffer (leave 1000 for other uses)
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
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/btts_odds_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # API protection
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
            logger.error(f"Reached max_requests limit ({self.max_requests}). Stopping to protect API quota.")
            return False

        return True

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request to The Odds API with rate limiting."""
        if not self.api_key:
            raise ValueError("API key not set")

        if not self._check_rate_limit():
            raise RuntimeError(
                f"API rate limit protection triggered. "
                f"Made {self.requests_made} requests, {self.requests_remaining} remaining. "
                f"Set max_requests higher if you're sure you have quota."
            )

        params["apiKey"] = self.api_key
        url = f"{THE_ODDS_API_BASE}{endpoint}"

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Track usage
        self.requests_made += 1
        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")

        if remaining is not None:
            self.requests_remaining = int(remaining)

        logger.info(f"API requests: {used} used, {remaining} remaining (session: {self.requests_made})")

        return response.json()

    def get_available_sports(self) -> List[Dict]:
        """Get list of available sports/leagues."""
        return self._make_request("/sports", {})

    def get_current_odds(
        self,
        league: str,
        regions: str = "uk,eu,us,au",
        bookmakers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get current BTTS odds for upcoming matches using event-odds endpoint.

        Args:
            league: League name (e.g., "premier_league")
            regions: Bookmaker regions (uk, us, eu, au)
            bookmakers: Specific bookmakers to include (optional)

        Returns:
            DataFrame with match info and BTTS odds
        """
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            raise ValueError(f"Unknown league: {league}. Available: {list(SPORT_KEYS.keys())}")

        try:
            events = self._make_request(f"/sports/{sport_key}/events", {})
        except requests.HTTPError as e:
            logger.error(f"Failed to fetch events: {e}")
            return pd.DataFrame()

        if not events:
            return pd.DataFrame()

        logger.info(f"Found {len(events)} upcoming events for {league}")

        matches = []
        for event in events:
            event_id = event.get("id")
            params = {
                "regions": regions,
                "markets": "btts",
                "oddsFormat": "decimal"
            }

            if bookmakers:
                params["bookmakers"] = ",".join(bookmakers)

            try:
                event_data = self._make_request(f"/sports/{sport_key}/events/{event_id}/odds", params)
            except requests.HTTPError as e:
                logger.warning(f"Failed to fetch odds for event {event_id}: {e}")
                continue

            match = {
                "event_id": event_id,
                "sport": sport_key,
                "commence_time": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
            }

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

            if btts_yes_odds:
                match["btts_yes_avg"] = np.mean(btts_yes_odds)
                match["btts_yes_max"] = max(btts_yes_odds)
                match["btts_yes_min"] = min(btts_yes_odds)
            if btts_no_odds:
                match["btts_no_avg"] = np.mean(btts_no_odds)
                match["btts_no_max"] = max(btts_no_odds)
                match["btts_no_min"] = min(btts_no_odds)

            if btts_yes_odds or btts_no_odds:
                match["odds_source"] = ODDS_SOURCE_REAL
                matches.append(match)

        df = pd.DataFrame(matches)

        if not df.empty:
            cache_file = self.cache_dir / f"{league}_btts_current.csv"
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached {len(df)} matches with BTTS odds to {cache_file}")

        return df

    def get_historical_odds(
        self,
        league: str,
        date: str,
        regions: str = "uk,eu,us"
    ) -> pd.DataFrame:
        """
        Get historical BTTS odds for a specific date.

        Args:
            league: League name
            date: Date in YYYY-MM-DD format
            regions: Bookmaker regions

        Returns:
            DataFrame with historical BTTS odds
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
                        "markets": "btts",
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

            if btts_yes_odds:
                match["btts_yes_avg"] = np.mean(btts_yes_odds)
                match["btts_yes_max"] = max(btts_yes_odds)
                match["btts_yes_min"] = min(btts_yes_odds)
            if btts_no_odds:
                match["btts_no_avg"] = np.mean(btts_no_odds)
                match["btts_no_max"] = max(btts_no_odds)
                match["btts_no_min"] = min(btts_no_odds)

            if btts_yes_odds or btts_no_odds:
                match["odds_source"] = ODDS_SOURCE_REAL
                matches.append(match)

        return pd.DataFrame(matches)

    def get_historical_odds_range(
        self,
        league: str,
        start_date: str,
        end_date: str,
        regions: str = "uk,eu,us"
    ) -> pd.DataFrame:
        """
        Get historical BTTS odds for a date range.

        Args:
            league: League name
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            regions: Bookmaker regions

        Returns:
            DataFrame with historical BTTS odds
        """
        from datetime import datetime, timedelta

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_odds = []
        current = start

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            logger.info(f"Fetching BTTS odds for {league} on {date_str}")

            df = self.get_historical_odds(league, date_str, regions)
            if not df.empty:
                all_odds.append(df)

            current += timedelta(days=1)

        if all_odds:
            combined = pd.concat(all_odds, ignore_index=True)
            combined = combined.drop_duplicates(subset=["event_id"])
            return combined

        return pd.DataFrame()

    def estimate_historical_odds(
        self,
        matches_df: pd.DataFrame,
        btts_yes_col: str = "btts",
        use_market_efficiency: bool = True
    ) -> pd.DataFrame:
        """
        Estimate BTTS odds for historical matches based on market patterns.

        Args:
            matches_df: DataFrame with match data including BTTS outcome
            btts_yes_col: Column name for BTTS outcome (1=Yes, 0=No)
            use_market_efficiency: If True, adjust odds based on actual BTTS rate

        Returns:
            DataFrame with estimated BTTS odds
        """
        df = matches_df.copy()

        base_yes = DEFAULT_BTTS_YES_ODDS
        base_no = DEFAULT_BTTS_NO_ODDS

        if use_market_efficiency and btts_yes_col in df.columns:
            btts_rate = df[btts_yes_col].mean()

            implied_yes = 1 / base_yes
            implied_no = 1 / base_no
            total_implied = implied_yes + implied_no

            true_yes = btts_rate
            true_no = 1 - btts_rate

            margin = total_implied - 1
            adj_yes = 1 / (true_yes + margin * true_yes / (true_yes + true_no))
            adj_no = 1 / (true_no + margin * true_no / (true_yes + true_no))

            logger.info(f"BTTS rate: {btts_rate:.1%}")
            logger.info(f"Estimated odds: Yes={adj_yes:.2f}, No={adj_no:.2f}")

            df["btts_yes_odds"] = adj_yes
            df["btts_no_odds"] = adj_no
        else:
            df["btts_yes_odds"] = base_yes
            df["btts_no_odds"] = base_no

        if "home_goals_scored_ema" in df.columns and "away_goals_scored_ema" in df.columns:
            attack_factor = (df["home_goals_scored_ema"] + df["away_goals_scored_ema"]) / 3
            attack_factor = attack_factor.clip(0.5, 1.5)

            df["btts_yes_odds"] = df["btts_yes_odds"] / attack_factor.clip(0.9, 1.1)
            df["btts_no_odds"] = df["btts_no_odds"] * attack_factor.clip(0.9, 1.1)

        # Mark as estimated odds
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


def add_btts_odds_to_features(
    features_df: pd.DataFrame,
    btts_col: str = "btts",
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Add BTTS odds to existing features DataFrame.

    Args:
        features_df: Features DataFrame with BTTS outcome
        btts_col: Column name for BTTS outcome
        output_path: Where to save result (optional)

    Returns:
        Features with BTTS odds added
    """
    loader = BTTSOddsLoader()
    result = loader.estimate_historical_odds(features_df, btts_col)

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = BTTSOddsLoader()

    status = loader.check_api_status()
    print(f"API Status: {json.dumps(status, indent=2)}")

    if status.get("status") == "ok":
        print("\nFetching Premier League BTTS odds...")
        odds = loader.get_current_odds("premier_league")
        if not odds.empty:
            print(odds[["home_team", "away_team", "btts_yes_avg", "btts_no_avg"]].head())
        else:
            print("No upcoming matches with BTTS odds")
    else:
        print("\nUsing estimated odds (no API key)")
        sample = pd.DataFrame({
            "btts": [1, 0, 1, 1, 0],
            "home_goals_scored_ema": [1.5, 1.0, 2.0, 1.8, 0.8],
            "away_goals_scored_ema": [1.2, 0.8, 1.5, 1.3, 0.5]
        })
        result = loader.estimate_historical_odds(sample)
        print(result[["btts", "btts_yes_odds", "btts_no_odds"]])
