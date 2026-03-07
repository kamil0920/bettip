"""
Base class for niche market odds loaders (The Odds API).

Shared infrastructure for BTTS, Cards, Corners, and Shots odds loaders.
Each subclass only needs to implement market-specific parsing and estimation.
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
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

ODDS_SOURCE_REAL = "real"
ODDS_SOURCE_ESTIMATED = "estimated"


class BaseOddsLoader(ABC):
    """Base class for niche market odds loaders using The Odds API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_requests: int = 19000,
        warn_at_remaining: int = 1000,
    ):
        self.api_key = api_key or THE_ODDS_API_KEY
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(f"data/{self._market_name}_odds_cache")
        )
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

    @property
    @abstractmethod
    def _market_name(self) -> str:
        """Short market name for cache dirs and logging (e.g., 'cards', 'btts')."""

    @property
    @abstractmethod
    def _api_market_key(self) -> str:
        """The Odds API market key (e.g., 'alternate_totals_cards', 'btts')."""

    @abstractmethod
    def _parse_event_odds(
        self, event_data: Dict, target_line: Optional[float] = None
    ) -> Optional[Dict]:
        """Parse odds from event data. Returns dict of odds columns or None."""

    @abstractmethod
    def _apply_estimation_adjustments(
        self, df: pd.DataFrame, target_line: Optional[float] = None
    ) -> pd.DataFrame:
        """Apply market-specific adjustments to estimated odds."""

    @property
    @abstractmethod
    def _default_over_odds(self) -> float:
        """Default over/yes odds when real odds unavailable."""

    @property
    @abstractmethod
    def _default_under_odds(self) -> float:
        """Default under/no odds when real odds unavailable."""

    @property
    @abstractmethod
    def _over_col(self) -> str:
        """Column name for over/yes odds (e.g., 'cards_over_odds', 'btts_yes_odds')."""

    @property
    @abstractmethod
    def _under_col(self) -> str:
        """Column name for under/no odds (e.g., 'cards_under_odds', 'btts_no_odds')."""

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

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request to The Odds API with rate limiting."""
        if not self.api_key:
            raise ValueError("API key not set")

        if not self._check_rate_limit():
            raise RuntimeError(
                f"API rate limit protection triggered. "
                f"Made {self.requests_made} requests, "
                f"{self.requests_remaining} remaining."
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

    def get_current_odds(
        self,
        league: str,
        target_line: Optional[float] = None,
        regions: str = "uk,eu",
        bookmakers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get current odds for upcoming matches."""
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            raise ValueError(
                f"Unknown league: {league}. Available: {list(SPORT_KEYS.keys())}"
            )

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
                "markets": self._api_market_key,
                "oddsFormat": "decimal",
            }

            if bookmakers:
                params["bookmakers"] = ",".join(bookmakers)

            try:
                event_data = self._make_request(
                    f"/sports/{sport_key}/events/{event_id}/odds", params
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

            odds_data = self._parse_event_odds(event_data, target_line)

            if odds_data:
                match.update(odds_data)
                match["odds_source"] = ODDS_SOURCE_REAL
                matches.append(match)

        df = pd.DataFrame(matches)

        if not df.empty:
            cache_file = self.cache_dir / f"{league}_{self._market_name}_current.csv"
            df.to_csv(cache_file, index=False)
            logger.info(
                f"Cached {len(df)} matches with {self._market_name} odds to {cache_file}"
            )

        return df

    def get_historical_odds(
        self,
        league: str,
        date: str,
        target_line: Optional[float] = None,
        regions: str = "uk,eu",
    ) -> pd.DataFrame:
        """Get historical odds for a specific date."""
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            raise ValueError(f"Unknown league: {league}")

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
            try:
                odds_response = self._make_request(
                    f"/historical/sports/{sport_key}/events/{event_id}/odds",
                    {
                        "regions": regions,
                        "markets": self._api_market_key,
                        "oddsFormat": "decimal",
                        "date": f"{date}T12:00:00Z",
                    },
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

            odds_data = self._parse_event_odds(event_data, target_line)
            if odds_data:
                match.update(odds_data)
                match["odds_source"] = ODDS_SOURCE_REAL
                matches.append(match)

        return pd.DataFrame(matches)

    def get_historical_odds_range(
        self,
        league: str,
        start_date: str,
        end_date: str,
        regions: str = "uk,eu,us",
    ) -> pd.DataFrame:
        """Get historical odds for a date range."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_odds = []
        current = start

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            logger.info(
                f"Fetching {self._market_name} odds for {league} on {date_str}"
            )

            df = self.get_historical_odds(league, date_str, regions=regions)
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
        outcome_col: str = "",
        target_line: Optional[float] = None,
        use_market_efficiency: bool = True,
    ) -> pd.DataFrame:
        """Estimate odds for historical matches based on market patterns."""
        df = matches_df.copy()

        base_over = self._default_over_odds
        base_under = self._default_under_odds

        if use_market_efficiency and outcome_col and outcome_col in df.columns:
            if target_line is not None:
                over_rate = (df[outcome_col] > target_line).mean()
            else:
                over_rate = df[outcome_col].mean()

            implied_over = 1 / base_over
            implied_under = 1 / base_under
            total_implied = implied_over + implied_under
            margin = total_implied - 1

            true_over = over_rate
            true_under = 1 - over_rate

            if true_over > 0 and true_under > 0:
                adj_over = 1 / (
                    true_over + margin * true_over / (true_over + true_under)
                )
                adj_under = 1 / (
                    true_under + margin * true_under / (true_over + true_under)
                )

                logger.info(
                    f"{self._market_name} over rate: {over_rate:.1%}"
                )
                logger.info(
                    f"Estimated odds: Over={adj_over:.2f}, Under={adj_under:.2f}"
                )

                df[self._over_col] = adj_over
                df[self._under_col] = adj_under
            else:
                df[self._over_col] = base_over
                df[self._under_col] = base_under
        else:
            df[self._over_col] = base_over
            df[self._under_col] = base_under

        df = self._apply_estimation_adjustments(df, target_line)
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
                timeout=10,
            )
            response.raise_for_status()

            return {
                "status": "ok",
                "requests_used": response.headers.get("x-requests-used", "?"),
                "requests_remaining": response.headers.get(
                    "x-requests-remaining", "?"
                ),
                "sports_available": len(response.json()),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


class TotalsOddsLoader(BaseOddsLoader, ABC):
    """Base for over/under totals markets (cards, corners, shots).

    Adds shared totals parsing: extracts all available lines from bookmaker
    data, finds closest to target, and produces over/under columns.
    """

    @property
    @abstractmethod
    def _common_lines(self) -> List[float]:
        """Common lines offered by bookmakers (e.g., [3.5, 4.5, 5.5])."""

    @property
    @abstractmethod
    def _default_line(self) -> float:
        """Default target line (e.g., 4.5 for cards, 9.5 for corners)."""

    @property
    @abstractmethod
    def _col_prefix(self) -> str:
        """Column prefix (e.g., 'cards', 'corners', 'shots')."""

    def _parse_event_odds(
        self, event_data: Dict, target_line: Optional[float] = None
    ) -> Optional[Dict]:
        """Parse totals odds, finding closest line to target."""
        if target_line is None:
            target_line = self._default_line

        all_lines: Dict[float, Dict[str, list]] = {}

        for bookmaker in event_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == self._api_market_key:
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
        prefix = self._col_prefix

        result = {
            f"{prefix}_line": closest_line,
            "available_lines": available_lines,
        }

        line_odds = all_lines[closest_line]
        if line_odds["over"]:
            result[f"{prefix}_over_avg"] = np.mean(line_odds["over"])
            result[f"{prefix}_over_max"] = max(line_odds["over"])
            result[f"{prefix}_over_min"] = min(line_odds["over"])
        if line_odds["under"]:
            result[f"{prefix}_under_avg"] = np.mean(line_odds["under"])
            result[f"{prefix}_under_max"] = max(line_odds["under"])
            result[f"{prefix}_under_min"] = min(line_odds["under"])

        for line in self._common_lines:
            if line in all_lines:
                lo = all_lines[line]
                line_str = str(line).replace(".", "_")
                if lo["over"]:
                    result[f"{prefix}_over_{line_str}_avg"] = np.mean(lo["over"])
                if lo["under"]:
                    result[f"{prefix}_under_{line_str}_avg"] = np.mean(lo["under"])

        return result

    def estimate_historical_odds(
        self,
        matches_df: pd.DataFrame,
        outcome_col: str = "",
        target_line: Optional[float] = None,
        use_market_efficiency: bool = True,
    ) -> pd.DataFrame:
        """Estimate odds, adding line metadata."""
        if target_line is None:
            target_line = self._default_line

        df = super().estimate_historical_odds(
            matches_df, outcome_col, target_line, use_market_efficiency
        )
        df[f"{self._col_prefix}_line"] = target_line
        return df
