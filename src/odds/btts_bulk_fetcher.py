"""
Optimized BTTS Odds Bulk Fetcher

Efficiently fetches historical BTTS odds from The Odds API.
Designed to maximize matches per API request.

Usage:
    python -m src.odds.btts_bulk_fetcher --start-date 2024-08-01 --end-date 2024-12-31
"""
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from fuzzywuzzy import fuzz

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("THE_ODDS_API_KEY", "")
API_BASE = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "premier_league": "soccer_epl",
    "la_liga": "soccer_spain_la_liga",
    "serie_a": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue_1": "soccer_france_ligue_one",
}


@dataclass
class FetchStats:
    """Track API usage statistics."""
    requests_made: int = 0
    matches_found: int = 0
    matches_with_btts: int = 0
    events_cached: int = 0
    requests_remaining: int = 0

    def efficiency(self) -> float:
        if self.requests_made == 0:
            return 0
        return self.matches_with_btts / self.requests_made


class BTTSBulkFetcher:
    """
    Efficiently fetch historical BTTS odds.

    Optimizations:
    1. Cache event IDs to avoid duplicate requests
    2. Only fetch for dates that exist in features
    3. Skip events we've already fetched
    4. Batch requests by actual match dates
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        features_path: Optional[Path] = None
    ):
        self.api_key = api_key or API_KEY
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/btts_odds_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.features_path = features_path or Path("data/03-features/features_all_5leagues_with_odds.csv")

        # Cache for fetched data
        self.event_cache: Dict[str, dict] = {}  # event_id -> event data
        self.fetched_events: Set[str] = set()  # event_ids we've already fetched odds for
        self.btts_odds: List[dict] = []  # collected BTTS odds

        self.stats = FetchStats()

        # Load existing cache
        self._load_cache()

    def _load_cache(self):
        """Load previously fetched data from cache."""
        cache_file = self.cache_dir / "btts_fetch_cache.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                self.fetched_events = set(data.get("fetched_events", []))
                self.stats.events_cached = len(self.fetched_events)
                logger.info(f"Loaded {len(self.fetched_events)} cached event IDs")

        # Load existing BTTS odds
        odds_file = self.cache_dir / "btts_odds_all.csv"
        if odds_file.exists():
            df = pd.read_csv(odds_file)
            self.btts_odds = df.to_dict('records')
            logger.info(f"Loaded {len(self.btts_odds)} existing BTTS odds")

    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "btts_fetch_cache.json"
        with open(cache_file, 'w') as f:
            json.dump({
                "fetched_events": list(self.fetched_events),
                "last_updated": datetime.now().isoformat()
            }, f)

        # Save BTTS odds
        if self.btts_odds:
            df = pd.DataFrame(self.btts_odds)
            df.to_csv(self.cache_dir / "btts_odds_all.csv", index=False)
            logger.info(f"Saved {len(self.btts_odds)} BTTS odds to cache")

    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make API request with tracking."""
        params["apiKey"] = self.api_key
        url = f"{API_BASE}{endpoint}"

        response = requests.get(url, params=params, timeout=30)
        self.stats.requests_made += 1

        # Track remaining requests
        self.stats.requests_remaining = int(response.headers.get("x-requests-remaining", 0))

        if self.stats.requests_made % 100 == 0:
            logger.info(f"Requests: {self.stats.requests_made}, Remaining: {self.stats.requests_remaining}")

        response.raise_for_status()
        return response.json()

    def get_target_dates(self) -> List[Tuple[str, str]]:
        """
        Get list of (date, league) tuples that need BTTS odds.
        Only returns dates that exist in our features file.
        """
        if not self.features_path.exists():
            logger.error(f"Features file not found: {self.features_path}")
            return []

        df = pd.read_csv(self.features_path)
        df['date'] = pd.to_datetime(df['date'])

        # Filter to dates that might have historical odds (recent matches)
        # The Odds API typically has ~1 year of historical data
        cutoff = datetime.now() - timedelta(days=365)
        recent = df[df['date'] >= cutoff]

        # Get unique date-league combinations
        # We need to infer league from team names or other columns
        targets = []
        for date in recent['date'].dt.date.unique():
            date_str = date.strftime("%Y-%m-%d")
            # Add all leagues for each date
            for league in SPORT_KEYS.keys():
                targets.append((date_str, league))

        logger.info(f"Found {len(targets)} date-league combinations to fetch")
        return sorted(targets)

    def fetch_events_for_date(self, date: str, league: str) -> List[dict]:
        """
        Fetch events for a specific date and league.
        Returns only events that match the exact date.
        """
        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            return []

        try:
            response = self._make_request(
                f"/historical/sports/{sport_key}/events",
                {"date": f"{date}T12:00:00Z"}
            )
        except requests.HTTPError as e:
            logger.warning(f"Failed to fetch events for {league} on {date}: {e}")
            return []

        events = response.get("data", [])

        # Filter to events that actually start on this date
        filtered = []
        for event in events:
            commence = event.get("commence_time", "")
            if commence.startswith(date):
                filtered.append(event)
                # Cache event info
                self.event_cache[event["id"]] = event

        return filtered

    def fetch_btts_odds_for_event(self, event_id: str, date: str, league: str) -> Optional[dict]:
        """
        Fetch BTTS odds for a single event.
        Returns None if no BTTS odds available.
        """
        if event_id in self.fetched_events:
            return None  # Already fetched

        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            return None

        try:
            response = self._make_request(
                f"/historical/sports/{sport_key}/events/{event_id}/odds",
                {
                    "regions": "uk,eu,us",
                    "markets": "btts",
                    "oddsFormat": "decimal",
                    "date": f"{date}T12:00:00Z"
                }
            )
        except requests.HTTPError:
            self.fetched_events.add(event_id)
            return None

        self.fetched_events.add(event_id)

        event_data = response.get("data", {})
        bookmakers = event_data.get("bookmakers", [])

        # Extract BTTS odds
        btts_yes = []
        btts_no = []

        for bm in bookmakers:
            for market in bm.get("markets", []):
                if market.get("key") == "btts":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Yes":
                            btts_yes.append(outcome.get("price"))
                        elif outcome.get("name") == "No":
                            btts_no.append(outcome.get("price"))

        if not btts_yes and not btts_no:
            return None

        # Get event info from cache
        event_info = self.event_cache.get(event_id, {})

        return {
            "event_id": event_id,
            "date": date,
            "league": league,
            "home_team": event_info.get("home_team", ""),
            "away_team": event_info.get("away_team", ""),
            "commence_time": event_info.get("commence_time", ""),
            "btts_yes_avg": np.mean(btts_yes) if btts_yes else None,
            "btts_yes_max": max(btts_yes) if btts_yes else None,
            "btts_yes_min": min(btts_yes) if btts_yes else None,
            "btts_no_avg": np.mean(btts_no) if btts_no else None,
            "btts_no_max": max(btts_no) if btts_no else None,
            "btts_no_min": min(btts_no) if btts_no else None,
            "num_bookmakers": len(btts_yes)
        }

    def fetch_all(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_requests: int = 19000,
        save_interval: int = 500
    ) -> pd.DataFrame:
        """
        Fetch BTTS odds for all target dates.

        Args:
            start_date: Start date (YYYY-MM-DD), defaults to 1 year ago
            end_date: End date (YYYY-MM-DD), defaults to today
            max_requests: Stop after this many requests (leave buffer)
            save_interval: Save cache every N requests

        Returns:
            DataFrame with BTTS odds
        """
        # Check API status first
        try:
            status = self._make_request("/sports", {})
            logger.info(f"API connected. Remaining requests: {self.stats.requests_remaining}")
        except Exception as e:
            logger.error(f"API connection failed: {e}")
            return pd.DataFrame()

        if self.stats.requests_remaining < 100:
            logger.error(f"Not enough API requests remaining: {self.stats.requests_remaining}")
            return pd.DataFrame()

        # Get target dates from features
        all_targets = self.get_target_dates()

        # Filter by date range
        if start_date:
            all_targets = [(d, l) for d, l in all_targets if d >= start_date]
        if end_date:
            all_targets = [(d, l) for d, l in all_targets if d <= end_date]

        logger.info(f"Processing {len(all_targets)} date-league combinations")
        logger.info(f"Max requests: {max_requests}, Current remaining: {self.stats.requests_remaining}")

        processed_dates = set()

        for date, league in all_targets:
            # Check if we should stop
            if self.stats.requests_made >= max_requests:
                logger.warning(f"Reached max requests limit ({max_requests})")
                break

            if self.stats.requests_remaining < 100:
                logger.warning("Low on API requests, stopping")
                break

            # Fetch events for this date-league
            events = self.fetch_events_for_date(date, league)
            self.stats.matches_found += len(events)

            # Fetch BTTS odds for each event
            for event in events:
                event_id = event["id"]

                if event_id in self.fetched_events:
                    continue

                odds = self.fetch_btts_odds_for_event(event_id, date, league)

                if odds:
                    self.btts_odds.append(odds)
                    self.stats.matches_with_btts += 1

                # Save periodically
                if self.stats.requests_made % save_interval == 0:
                    self._save_cache()
                    logger.info(
                        f"Progress: {self.stats.requests_made} requests, "
                        f"{self.stats.matches_with_btts} matches with BTTS, "
                        f"efficiency: {self.stats.efficiency():.2%}"
                    )

        # Final save
        self._save_cache()

        # Log final stats
        logger.info("=" * 50)
        logger.info("FETCH COMPLETE")
        logger.info(f"Total requests: {self.stats.requests_made}")
        logger.info(f"Matches found: {self.stats.matches_found}")
        logger.info(f"Matches with BTTS: {self.stats.matches_with_btts}")
        logger.info(f"Efficiency: {self.stats.efficiency():.2%}")
        logger.info(f"Requests remaining: {self.stats.requests_remaining}")
        logger.info("=" * 50)

        return pd.DataFrame(self.btts_odds)

    def check_status(self) -> dict:
        """Check API status and cache state."""
        try:
            self._make_request("/sports", {})
            api_ok = True
        except:
            api_ok = False

        return {
            "api_connected": api_ok,
            "requests_remaining": self.stats.requests_remaining,
            "cached_events": len(self.fetched_events),
            "btts_odds_collected": len(self.btts_odds),
            "efficiency": self.stats.efficiency()
        }


def merge_btts_with_features(
    btts_path: Path = Path("data/btts_odds_cache/btts_odds_all.csv"),
    features_path: Path = Path("data/03-features/features_all_5leagues_with_odds.csv"),
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Merge fetched BTTS odds with features file.
    Uses fuzzy matching for team names.
    """
    if not btts_path.exists():
        logger.error(f"BTTS odds file not found: {btts_path}")
        return pd.DataFrame()

    btts_df = pd.read_csv(btts_path)
    features_df = pd.read_csv(features_path)

    logger.info(f"BTTS odds: {len(btts_df)} matches")
    logger.info(f"Features: {len(features_df)} matches")

    # Normalize dates
    features_df['date'] = pd.to_datetime(features_df['date']).dt.tz_localize(None)
    btts_df['date'] = pd.to_datetime(btts_df['date']).dt.tz_localize(None)

    # Normalize team names for matching
    def normalize(name):
        name = str(name).lower().strip()
        for suffix in [' fc', ' united', ' city', ' hotspur', ' wanderers', ' albion']:
            name = name.replace(suffix, '')
        return name

    # Create match index
    matched = 0
    btts_map = {}

    for _, btts_row in btts_df.iterrows():
        btts_date = btts_row['date'].date()
        btts_home = normalize(btts_row['home_team'])
        btts_away = normalize(btts_row['away_team'])

        # Find matching feature row
        same_date = features_df[features_df['date'].dt.date == btts_date]

        for idx, feat_row in same_date.iterrows():
            feat_home = normalize(feat_row['home_team_name'])
            feat_away = normalize(feat_row['away_team_name'])

            # Check match
            home_ok = fuzz.ratio(btts_home, feat_home) >= 75 or btts_home in feat_home or feat_home in btts_home
            away_ok = fuzz.ratio(btts_away, feat_away) >= 75 or btts_away in feat_away or feat_away in btts_away

            if home_ok and away_ok:
                btts_map[idx] = {
                    'btts_yes_avg': btts_row['btts_yes_avg'],
                    'btts_no_avg': btts_row['btts_no_avg'],
                    'btts_yes_max': btts_row.get('btts_yes_max'),
                    'btts_no_max': btts_row.get('btts_no_max'),
                }
                matched += 1
                break

    logger.info(f"Matched {matched} BTTS odds to features")

    # Apply matches
    for idx, odds in btts_map.items():
        for col, val in odds.items():
            if pd.notna(val):
                features_df.loc[idx, col] = val

    # Fill missing with estimates
    mask = features_df['btts_yes_avg'].isna()
    if 'home_goals_scored_ema' in features_df.columns:
        attack = (features_df['home_goals_scored_ema'].fillna(1.3) +
                  features_df['away_goals_scored_ema'].fillna(1.1)) / 2.4
        attack = attack.clip(0.8, 1.2)
        features_df.loc[mask, 'btts_yes_avg'] = 1.80 / attack[mask]
        features_df.loc[mask, 'btts_no_avg'] = 2.00 * attack[mask]
    else:
        features_df.loc[mask, 'btts_yes_avg'] = 1.80
        features_df.loc[mask, 'btts_no_avg'] = 2.00

    features_df['btts_yes_max'] = features_df['btts_yes_max'].fillna(features_df['btts_yes_avg'] * 1.05)
    features_df['btts_no_max'] = features_df['btts_no_max'].fillna(features_df['btts_no_avg'] * 1.05)

    # Save
    if output_path is None:
        output_path = features_path

    features_df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    real_btts = matched
    estimated = len(features_df) - matched
    logger.info(f"Real BTTS odds: {real_btts} ({100*real_btts/len(features_df):.1f}%)")
    logger.info(f"Estimated: {estimated} ({100*estimated/len(features_df):.1f}%)")

    return features_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch BTTS odds efficiently")
    parser.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--max-requests", type=int, default=19000, help="Max API requests")
    parser.add_argument("--check-only", action="store_true", help="Only check status")
    parser.add_argument("--merge", action="store_true", help="Merge with features after fetch")

    args = parser.parse_args()

    fetcher = BTTSBulkFetcher()

    if args.check_only:
        status = fetcher.check_status()
        print(json.dumps(status, indent=2))
    else:
        df = fetcher.fetch_all(
            start_date=args.start_date,
            end_date=args.end_date,
            max_requests=args.max_requests
        )

        if args.merge and not df.empty:
            merge_btts_with_features()
