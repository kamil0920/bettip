"""
Historical Niche Odds Collection — Cards & Corners from The Odds API

Fetches historical per-line odds for cards and corners markets using a two-step
approach: (1) list events via /historical/sports/{sport}/events, (2) fetch
per-event odds via /historical/sports/{sport}/events/{id}/odds.

The bulk /historical/sports/{sport}/odds endpoint returns 422 for niche markets,
so per-event fetching is required.

Supports two market types:
- **totals** (default): alternate_totals_cards, alternate_totals_corners
- **spreads**: alternate_spreads_corners, alternate_spreads_cards (HC markets)

Designed for incremental backfill: resumable, cached, quota-aware.

Usage:
    # Last weekend (Fri-Sun) — totals (default)
    python scripts/collect_historical_niche_odds.py --last-weekend

    # Date range — spreads (HC markets)
    python scripts/collect_historical_niche_odds.py --start-date 2024-08-01 --end-date 2026-02-26 --markets spreads

    # All markets (totals + spreads)
    python scripts/collect_historical_niche_odds.py --start-date 2025-01-01 --end-date 2026-02-26 --markets all

    # Weekends only in date range
    python scripts/collect_historical_niche_odds.py --start-date 2025-01-01 --end-date 2026-02-26 --weekends-only

    # Budget control
    python scripts/collect_historical_niche_odds.py --last-weekend --max-credits 1000 --regions uk
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.odds.theodds_unified_loader import SPORT_KEYS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("THE_ODDS_API_KEY", "")
API_BASE = "https://api.the-odds-api.com/v4"

# Markets to fetch — only cards and corners have API coverage
TOTALS_MARKETS = "alternate_totals_cards,alternate_totals_corners"
SPREADS_MARKETS = "alternate_spreads_corners,alternate_spreads_cards"
ALL_MARKETS = f"{TOTALS_MARKETS},{SPREADS_MARKETS}"

# Map --markets CLI flag to API market strings
MARKET_PRESETS = {
    "totals": TOTALS_MARKETS,
    "spreads": SPREADS_MARKETS,
    "all": ALL_MARKETS,
}

# Our 10 active leagues (excluding mls, liga_mx, ekstraklasa — no API coverage)
ACTIVE_LEAGUES = [
    "premier_league",
    "la_liga",
    "serie_a",
    "bundesliga",
    "ligue_1",
    "eredivisie",
    "portuguese_liga",
    "scottish_premiership",
    "turkish_super_lig",
    "belgian_pro_league",
]

# Cost: historical events = 1 credit, historical per-event odds = 10 × regions × markets
CREDITS_PER_EVENTS_CALL = 1

OUTPUT_DIR = Path("data/historical_niche_odds")
PARQUET_PATH = OUTPUT_DIR / "niche_odds_historical.parquet"
CACHE_PATH = OUTPUT_DIR / "fetch_cache.json"


def load_cache() -> Dict:
    """Load fetch cache from disk."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {
        "fetched_pairs": [],
        "fetched_events": [],
        "total_credits_used": 0,
        "last_updated": None,
    }


def save_cache(cache: Dict) -> None:
    """Save fetch cache to disk."""
    cache["last_updated"] = datetime.now().isoformat()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def make_request(endpoint: str, params: dict) -> Tuple[Optional[dict], int]:
    """Make API request. Returns (json_data, requests_remaining)."""
    params["apiKey"] = API_KEY
    url = f"{API_BASE}{endpoint}"

    response = requests.get(url, params=params, timeout=30)
    remaining = int(response.headers.get("x-requests-remaining", -1))

    response.raise_for_status()
    return response.json(), remaining


def compute_credits_per_odds_call(regions: str, markets: str) -> int:
    """Credits per historical per-event odds call.

    Cost: 10 × regions × markets.
    """
    n_regions = len(regions.split(","))
    n_markets = len(markets.split(","))
    return 10 * n_regions * n_markets


def build_date_league_pairs(
    start_date: str,
    end_date: str,
    weekends_only: bool = False,
    leagues: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Build list of (date_str, league) pairs to fetch."""
    if leagues is None:
        leagues = ACTIVE_LEAGUES

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    pairs = []
    current = start
    while current <= end:
        if weekends_only and current.weekday() not in (4, 5, 6):
            current += timedelta(days=1)
            continue
        date_str = current.strftime("%Y-%m-%d")
        for league in leagues:
            if league in SPORT_KEYS:
                pairs.append((date_str, league))
        current += timedelta(days=1)

    return sorted(pairs)


def get_last_weekend_dates() -> Tuple[str, str]:
    """Get the date range for last weekend (Fri-Sun)."""
    today = datetime.now()
    days_since_sunday = (today.weekday() + 1) % 7
    if days_since_sunday == 0:
        days_since_sunday = 7
    last_sunday = today - timedelta(days=days_since_sunday)
    last_friday = last_sunday - timedelta(days=2)
    return last_friday.strftime("%Y-%m-%d"), last_sunday.strftime("%Y-%m-%d")


_TOTALS_KEYS = {"alternate_totals_cards", "alternate_totals_corners"}
_SPREADS_KEYS = {"alternate_spreads_corners", "alternate_spreads_cards"}
_ALL_NICHE_KEYS = _TOTALS_KEYS | _SPREADS_KEYS


def parse_event_odds(
    data: dict,
    event_info: dict,
    league: str,
    date_str: str,
) -> List[Dict]:
    """Parse per-event odds response into flat records.

    Handles both totals (Over/Under by name) and spreads (home/away team handicaps).
    Spread outcomes are normalized to home-team perspective:
    - Home team negative point → "over" (home favoured)
    - Away team positive point → "under" (away favoured)

    Args:
        data: Raw API response from /historical/sports/{sport}/events/{id}/odds.
        event_info: Event metadata (id, home_team, away_team, commence_time).
        league: Our league name.
        date_str: Date string YYYY-MM-DD.

    Returns:
        List of dicts, one per (market, line) combination.
    """
    event_data = data.get("data", {})
    bookmakers = event_data.get("bookmakers", [])
    records = []

    home_team = event_info.get("home_team", "")
    away_team = event_info.get("away_team", "")

    # Collect per-line odds across bookmakers
    # Key: (market_key, line) -> {"over": [prices], "under": [prices]}
    line_odds: Dict[Tuple[str, float], Dict[str, List[float]]] = {}

    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")
            if market_key not in _ALL_NICHE_KEYS:
                continue

            is_spread = market_key in _SPREADS_KEYS

            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                point = outcome.get("point")
                if price is None or point is None:
                    continue

                if is_spread:
                    # Spread: normalize to home-team perspective using abs(point)
                    abs_line = abs(float(point))
                    key = (market_key, abs_line)
                    if key not in line_odds:
                        line_odds[key] = {"over": [], "under": []}

                    if name == home_team:
                        if float(point) < 0:
                            line_odds[key]["over"].append(float(price))
                        else:
                            line_odds[key]["under"].append(float(price))
                    elif name == away_team:
                        if float(point) > 0:
                            line_odds[key]["under"].append(float(price))
                        else:
                            line_odds[key]["over"].append(float(price))
                else:
                    # Totals: Over/Under by name
                    line = float(point)
                    key = (market_key, line)
                    if key not in line_odds:
                        line_odds[key] = {"over": [], "under": []}

                    if "Over" in name:
                        line_odds[key]["over"].append(float(price))
                    elif "Under" in name:
                        line_odds[key]["under"].append(float(price))

    # Convert to flat records
    for (market_key, line), odds in line_odds.items():
        # Map API market key to our stat name
        if market_key == "alternate_totals_cards":
            stat = "cards"
        elif market_key == "alternate_totals_corners":
            stat = "corners"
        elif market_key == "alternate_spreads_corners":
            stat = "cornershc"
        elif market_key == "alternate_spreads_cards":
            stat = "cardshc"
        else:
            continue

        over_prices = odds["over"]
        under_prices = odds["under"]

        if not over_prices and not under_prices:
            continue

        records.append(
            {
                "event_id": event_info["id"],
                "date": date_str,
                "commence_time": event_info.get("commence_time", ""),
                "league": league,
                "sport_key": SPORT_KEYS[league],
                "home_team": home_team,
                "away_team": away_team,
                "market": stat,
                "line": line,
                "over_avg": float(np.mean(over_prices)) if over_prices else None,
                "under_avg": float(np.mean(under_prices)) if under_prices else None,
                "over_max": float(max(over_prices)) if over_prices else None,
                "under_max": float(max(under_prices)) if under_prices else None,
                "num_bookmakers": max(len(over_prices), len(under_prices)),
            }
        )

    return records


def fetch_historical_niche_odds(
    pairs: List[Tuple[str, str]],
    regions: str = "us",
    max_credits: int = 5000,
    save_interval: int = 20,
    markets: str = TOTALS_MARKETS,
) -> pd.DataFrame:
    """Fetch historical niche odds for given (date, league) pairs.

    Two-step approach per (date, league):
    1. GET /historical/sports/{sport}/events?date=... → list events (1 credit)
    2. For each event: GET /historical/.../events/{id}/odds → niche odds

    The timestamp for per-event odds uses 1 hour before commence_time to get
    pre-match odds (the API requires a timestamp when odds were live).

    Args:
        pairs: List of (date_str, league) to fetch.
        regions: Bookmaker regions (default "us").
        max_credits: Stop after this many credits used.
        save_interval: Save cache every N odds API calls.
        markets: Comma-separated API market keys to fetch.

    Returns:
        DataFrame with all collected odds.
    """
    if not API_KEY:
        logger.error("THE_ODDS_API_KEY not set in .env")
        return pd.DataFrame()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cache and existing data
    cache = load_cache()
    # Pair cache is market-aware: "date|league|markets_string"
    # Legacy "date|league" entries are treated as fetched for TOTALS_MARKETS only.
    raw_pairs = cache.get("fetched_pairs", [])
    fetched_pairs_set: Set[str] = set()
    for entry in raw_pairs:
        if len(entry) == 3:
            d, l, m = entry
            fetched_pairs_set.add(f"{d}|{l}|{m}")
        else:
            d, l = entry
            fetched_pairs_set.add(f"{d}|{l}|{TOTALS_MARKETS}")
    # Event cache is market-aware: "event_id|markets_string" to distinguish
    # totals-only fetches from spreads fetches. Legacy plain event IDs are
    # treated as fetched for TOTALS_MARKETS only.
    raw_events = cache.get("fetched_events", [])
    fetched_events_set: Set[str] = set()
    for e in raw_events:
        if "|" in e:
            fetched_events_set.add(e)
        else:
            # Legacy: plain event ID → assume fetched for totals only
            fetched_events_set.add(f"{e}|{TOTALS_MARKETS}")
    credits_used = cache.get("total_credits_used", 0)
    credits_per_odds = compute_credits_per_odds_call(regions, markets)

    # Load existing records
    all_records: List[Dict] = []
    if PARQUET_PATH.exists():
        existing_df = pd.read_parquet(PARQUET_PATH)
        all_records = existing_df.to_dict("records")
        logger.info(f"Loaded {len(all_records)} existing odds records")

    # Filter out already-fetched pairs (market-aware)
    new_pairs = [
        (d, l) for d, l in pairs if f"{d}|{l}|{markets}" not in fetched_pairs_set
    ]
    logger.info(
        f"{len(new_pairs)} new pairs to fetch "
        f"({len(pairs) - len(new_pairs)} already cached)"
    )

    if not new_pairs:
        logger.info("Nothing to fetch — all pairs already cached")
        return pd.DataFrame(all_records) if all_records else pd.DataFrame()

    # Check API connectivity
    try:
        _, remaining = make_request("/sports", {})
        logger.info(f"API connected. Remaining requests: {remaining}")
    except Exception as e:
        logger.error(f"API connection failed: {e}")
        return pd.DataFrame()

    event_calls = 0
    odds_calls = 0
    new_records_count = 0

    for date_str, league in new_pairs:
        # Budget check (reserve credits for at least 1 events + 1 odds call)
        min_needed = CREDITS_PER_EVENTS_CALL + credits_per_odds
        if credits_used + min_needed > max_credits:
            logger.warning(
                f"Budget limit reached: {credits_used} credits used, "
                f"need {min_needed} for next pair, limit is {max_credits}"
            )
            break

        sport_key = SPORT_KEYS.get(league)
        if not sport_key:
            continue

        # Step 1: Get events for this date+league
        try:
            events_data, remaining = make_request(
                f"/historical/sports/{sport_key}/events",
                {"date": f"{date_str}T12:00:00Z"},
            )
        except requests.HTTPError as e:
            logger.warning(f"Events failed {league} {date_str}: {e}")
            fetched_pairs_set.add(f"{date_str}|{league}|{markets}")
            cache["fetched_pairs"].append([date_str, league, markets])
            continue

        event_calls += 1
        credits_used += CREDITS_PER_EVENTS_CALL

        events = events_data.get("data", [])
        # Filter to events on the target date
        day_events = [
            e for e in events
            if e.get("commence_time", "").startswith(date_str)
        ]

        if not day_events:
            logger.debug(f"  {league} {date_str}: no events")
            fetched_pairs_set.add(f"{date_str}|{league}|{markets}")
            cache["fetched_pairs"].append([date_str, league, markets])
            continue

        # Step 2: Fetch odds per event
        pair_records = 0
        for event in day_events:
            event_id = event.get("id", "")

            # Skip already-fetched events (market-aware cache key)
            event_cache_key = f"{event_id}|{markets}"
            if event_cache_key in fetched_events_set:
                continue

            # Budget check per event
            if credits_used + credits_per_odds > max_credits:
                logger.warning(f"Budget limit reached mid-pair at {credits_used} credits")
                break

            # Use 1 hour before commence_time for pre-match odds
            commence = event.get("commence_time", "")
            try:
                ct = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                odds_ts = (ct - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, TypeError):
                odds_ts = f"{date_str}T12:00:00Z"

            try:
                odds_data, remaining = make_request(
                    f"/historical/sports/{sport_key}/events/{event_id}/odds",
                    {
                        "date": odds_ts,
                        "regions": regions,
                        "markets": markets,
                        "oddsFormat": "decimal",
                    },
                )
            except requests.HTTPError as e:
                logger.debug(f"  Odds failed {event_id}: {e}")
                fetched_events_set.add(event_cache_key)
                cache["fetched_events"].append(event_cache_key)
                continue

            odds_calls += 1
            credits_used += credits_per_odds

            # Parse odds
            records = parse_event_odds(odds_data, event, league, date_str)
            all_records.extend(records)
            new_records_count += len(records)
            pair_records += len(records)

            # Mark event as fetched (market-aware)
            fetched_events_set.add(event_cache_key)
            cache["fetched_events"].append(event_cache_key)
            cache["total_credits_used"] = credits_used

            # Save periodically
            if odds_calls % save_interval == 0 and odds_calls > 0:
                _save_results(all_records, cache)
                logger.info(
                    f"Progress: {event_calls} event calls, {odds_calls} odds calls, "
                    f"{credits_used} credits, {new_records_count} new records"
                )

            # Safety: stop if API quota low
            if 0 <= remaining < 100:
                logger.warning(f"API quota low ({remaining} remaining), stopping")
                break

        if pair_records:
            logger.info(
                f"  {league} {date_str}: {len(day_events)} events, "
                f"{pair_records} line-odds"
            )
        else:
            logger.info(f"  {league} {date_str}: {len(day_events)} events, no niche odds")

        # Mark pair as fetched (market-aware)
        fetched_pairs_set.add(f"{date_str}|{league}|{markets}")
        cache["fetched_pairs"].append([date_str, league, markets])

    # Final save
    _save_results(all_records, cache)

    logger.info("=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info(f"Event API calls:    {event_calls}")
    logger.info(f"Odds API calls:     {odds_calls}")
    logger.info(f"Credits used:       {credits_used}")
    logger.info(f"New records:        {new_records_count}")
    logger.info(f"Total records:      {len(all_records)}")
    logger.info(f"Output:             {PARQUET_PATH}")
    logger.info("=" * 60)

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def _save_results(records: List[Dict], cache: Dict) -> None:
    """Save records to parquet and cache to JSON."""
    if records:
        df = pd.DataFrame(records)
        # Deduplicate by (event_id, market, line)
        df = df.drop_duplicates(subset=["event_id", "market", "line"], keep="last")
        df.to_parquet(PARQUET_PATH, index=False)
    save_cache(cache)


def print_summary(df: pd.DataFrame) -> None:
    """Print summary of collected data."""
    if df.empty:
        print("\nNo data collected.")
        return

    print(f"\n--- Collection Summary ---")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique dates: {df['date'].nunique()}")
    print(f"Unique events: {df['event_id'].nunique()}")

    print(f"\nPer-market breakdown:")
    for market in sorted(df["market"].unique()):
        mdf = df[df["market"] == market]
        lines = sorted(mdf["line"].unique())
        print(
            f"  {market}: {len(mdf)} records, "
            f"{mdf['event_id'].nunique()} events, "
            f"lines: {lines}"
        )

    print(f"\nPer-league breakdown:")
    for league in sorted(df["league"].unique()):
        ldf = df[df["league"] == league]
        print(
            f"  {league}: {ldf['event_id'].nunique()} events, "
            f"{len(ldf)} records"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect historical niche odds (cards/corners) from The Odds API"
    )
    parser.add_argument(
        "--last-weekend",
        action="store_true",
        help="Fetch odds for last weekend (Fri-Sun)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--weekends-only",
        action="store_true",
        help="Only fetch for Fri/Sat/Sun dates",
    )
    parser.add_argument(
        "--max-credits",
        type=int,
        default=5000,
        help="Maximum API credits to use (default 5000)",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="us",
        help="Bookmaker regions (default 'us' — niche markets from US books). More regions = more credits.",
    )
    parser.add_argument(
        "--leagues",
        type=str,
        default=None,
        help="Comma-separated league names (default: all 10 active)",
    )
    parser.add_argument(
        "--markets",
        type=str,
        default="totals",
        choices=["totals", "spreads", "all"],
        help="Market type to fetch: totals (O/U lines), spreads (HC lines), or all (default: totals)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only show cache status, don't fetch",
    )

    args = parser.parse_args()

    if args.check_only:
        cache = load_cache()
        print(json.dumps(cache, indent=2))
        if PARQUET_PATH.exists():
            df = pd.read_parquet(PARQUET_PATH)
            print_summary(df)
        return

    # Determine date range
    if args.last_weekend:
        start_date, end_date = get_last_weekend_dates()
        logger.info(f"Last weekend: {start_date} to {end_date}")
    elif args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        parser.error("Specify --last-weekend or --start-date/--end-date")
        return

    leagues = args.leagues.split(",") if args.leagues else None
    api_markets = MARKET_PRESETS[args.markets]
    logger.info(f"Markets: {args.markets} → {api_markets}")

    # Build pairs
    pairs = build_date_league_pairs(
        start_date, end_date, weekends_only=args.weekends_only, leagues=leagues
    )
    logger.info(f"Built {len(pairs)} date-league pairs to consider")

    # Fetch
    df = fetch_historical_niche_odds(
        pairs,
        regions=args.regions,
        max_credits=args.max_credits,
        markets=api_markets,
    )

    print_summary(df)


if __name__ == "__main__":
    main()
