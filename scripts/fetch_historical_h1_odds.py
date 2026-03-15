"""Fetch historical H1 (first half) 1X2 odds from The Odds API.

Samples ~200 events across Big 5 leagues to build an empirical H1 odds
distribution. Budget: ~2,000 credits (10 credits/event/region/market).

Usage:
    python scripts/fetch_historical_h1_odds.py --dry-run   # check credit cost
    python scripts/fetch_historical_h1_odds.py              # fetch and save
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

API_BASE = "https://api.the-odds-api.com/v4"
API_KEY = os.getenv("THE_ODDS_API_KEY")

# Big 5 sport keys — spread events across leagues for diversity
LEAGUES = {
    "soccer_epl": "premier_league",
    "soccer_spain_la_liga": "la_liga",
    "soccer_italy_serie_a": "serie_a",
    "soccer_germany_bundesliga": "bundesliga",
    "soccer_france_ligue_one": "ligue_1",
}

# Sample dates: ~4 matchdays per league × 5 leagues = 20 matchdays × ~10 events = 200
# Pick Saturdays spread across 2024-25 season (within The Odds API historical window)
SAMPLE_DATES = [
    # EPL: 4 dates
    ("soccer_epl", "2025-01-18"),
    ("soccer_epl", "2025-03-15"),
    ("soccer_epl", "2025-09-27"),
    ("soccer_epl", "2025-11-22"),
    # La Liga: 4 dates
    ("soccer_spain_la_liga", "2025-01-25"),
    ("soccer_spain_la_liga", "2025-04-05"),
    ("soccer_spain_la_liga", "2025-10-04"),
    ("soccer_spain_la_liga", "2025-12-06"),
    # Serie A: 4 dates
    ("soccer_italy_serie_a", "2025-02-01"),
    ("soccer_italy_serie_a", "2025-04-12"),
    ("soccer_italy_serie_a", "2025-10-25"),
    ("soccer_italy_serie_a", "2025-12-20"),
    # Bundesliga: 4 dates
    ("soccer_germany_bundesliga", "2025-02-08"),
    ("soccer_germany_bundesliga", "2025-03-29"),
    ("soccer_germany_bundesliga", "2025-11-01"),
    ("soccer_germany_bundesliga", "2025-12-13"),
    # Ligue 1: 4 dates
    ("soccer_france_ligue_one", "2025-02-15"),
    ("soccer_france_ligue_one", "2025-04-19"),
    ("soccer_france_ligue_one", "2025-10-18"),
    ("soccer_france_ligue_one", "2025-11-29"),
]

OUTPUT_PATH = Path("data/sportmonks_odds/processed/h1_historical_odds.csv")


def make_request(endpoint: str, params: dict) -> dict:
    """Make API request with rate limit tracking."""
    params["apiKey"] = API_KEY
    url = f"{API_BASE}{endpoint}"
    response = requests.get(url, params=params, timeout=30)

    remaining = response.headers.get("x-requests-remaining", "?")
    used = response.headers.get("x-requests-used", "?")
    logger.info(f"  Credits: {used} used, {remaining} remaining")

    response.raise_for_status()
    return response.json()


def fetch_events_for_date(sport_key: str, date: str) -> list:
    """Fetch historical events list for a given date. Costs 1 credit."""
    try:
        resp = make_request(
            f"/historical/sports/{sport_key}/events",
            {"date": f"{date}T15:00:00Z"},
        )
        events = resp.get("data", [])
        logger.info(f"  {sport_key} {date}: {len(events)} events found")
        return events
    except requests.HTTPError as e:
        logger.warning(f"  Failed to fetch events for {sport_key} {date}: {e}")
        return []


def fetch_h1_odds_for_event(sport_key: str, event_id: str, date: str) -> dict:
    """Fetch h2h_h1 odds for a single event. Costs 10 credits."""
    try:
        resp = make_request(
            f"/historical/sports/{sport_key}/events/{event_id}/odds",
            {
                "regions": "uk,eu",
                "markets": "h2h_h1",
                "oddsFormat": "decimal",
                "date": f"{date}T15:00:00Z",
            },
        )
        return resp.get("data", {})
    except requests.HTTPError as e:
        logger.warning(f"  Failed to fetch h2h_h1 for event {event_id}: {e}")
        return {}


def parse_h1_odds(event_data: dict) -> dict:
    """Parse H1 1X2 odds from API response."""
    home_odds = []
    draw_odds = []
    away_odds = []

    home_team = event_data.get("home_team", "")
    away_team = event_data.get("away_team", "")

    for bookmaker in event_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") == "h2h_h1":
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    if price is None:
                        continue
                    if name == "Draw":
                        draw_odds.append(price)
                    elif name == home_team:
                        home_odds.append(price)
                    elif name == away_team:
                        away_odds.append(price)

    if not home_odds and not away_odds:
        return {}

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "n_bookmakers": len(home_odds),
    }
    if home_odds:
        result["h2h_h1_home_avg"] = np.mean(home_odds)
        result["h2h_h1_home_min"] = min(home_odds)
        result["h2h_h1_home_max"] = max(home_odds)
    if draw_odds:
        result["h2h_h1_draw_avg"] = np.mean(draw_odds)
    if away_odds:
        result["h2h_h1_away_avg"] = np.mean(away_odds)
        result["h2h_h1_away_min"] = min(away_odds)
        result["h2h_h1_away_max"] = max(away_odds)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only count events, don't fetch odds")
    parser.add_argument("--max-events", type=int, default=200, help="Max events to fetch H1 odds for")
    args = parser.parse_args()

    if not API_KEY:
        raise ValueError("THE_ODDS_API_KEY not set in .env")

    logger.info(f"Fetching historical H1 odds (max {args.max_events} events)")
    logger.info(f"Estimated cost: {args.max_events * 10 + len(SAMPLE_DATES)} credits")

    all_results = []
    total_events_found = 0
    events_fetched = 0

    for sport_key, date in SAMPLE_DATES:
        if events_fetched >= args.max_events:
            logger.info(f"Reached max events ({args.max_events}), stopping.")
            break

        league = LEAGUES[sport_key]
        logger.info(f"\n--- {league} {date} ---")

        # Step 1: Get events list (1 credit)
        events = fetch_events_for_date(sport_key, date)
        total_events_found += len(events)

        if args.dry_run:
            for ev in events:
                logger.info(f"  [DRY] {ev.get('home_team')} vs {ev.get('away_team')}")
            continue

        # Step 2: Fetch H1 odds for each event (10 credits each)
        for event in events:
            if events_fetched >= args.max_events:
                break

            event_id = event.get("id")
            home = event.get("home_team", "?")
            away = event.get("away_team", "?")
            commence = event.get("commence_time", "")

            logger.info(f"  Fetching H1 odds: {home} vs {away}")
            event_data = fetch_h1_odds_for_event(sport_key, event_id, date)

            if event_data:
                parsed = parse_h1_odds(event_data)
                if parsed:
                    parsed["league"] = league
                    parsed["date"] = date
                    parsed["event_id"] = event_id
                    parsed["commence_time"] = commence
                    all_results.append(parsed)
                    events_fetched += 1
                    logger.info(
                        f"    OK: home={parsed.get('h2h_h1_home_avg', '?'):.2f}, "
                        f"away={parsed.get('h2h_h1_away_avg', '?'):.2f}, "
                        f"bookmakers={parsed.get('n_bookmakers', 0)}"
                    )
                else:
                    logger.info(f"    No H1 odds available for this event")
                    events_fetched += 1  # still costs credits

            # Rate limiting: 1 second between requests
            time.sleep(1.0)

    if args.dry_run:
        logger.info(f"\n=== DRY RUN SUMMARY ===")
        logger.info(f"Total events found: {total_events_found}")
        logger.info(f"Estimated credit cost: {total_events_found * 10 + len(SAMPLE_DATES)}")
        return

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"\n=== RESULTS ===")
        logger.info(f"Events with H1 odds: {len(df)}/{events_fetched}")
        logger.info(f"Saved to: {OUTPUT_PATH}")
        logger.info(f"\nH1 away odds distribution:")
        if "h2h_h1_away_avg" in df.columns:
            desc = df["h2h_h1_away_avg"].describe()
            logger.info(f"  mean={desc['mean']:.2f}, std={desc['std']:.2f}")
            logger.info(f"  min={desc['min']:.2f}, P25={desc['25%']:.2f}, "
                       f"median={desc['50%']:.2f}, P75={desc['75%']:.2f}, max={desc['max']:.2f}")
        logger.info(f"\nH1 home odds distribution:")
        if "h2h_h1_home_avg" in df.columns:
            desc = df["h2h_h1_home_avg"].describe()
            logger.info(f"  mean={desc['mean']:.2f}, std={desc['std']:.2f}")
            logger.info(f"  min={desc['min']:.2f}, P25={desc['25%']:.2f}, "
                       f"median={desc['50%']:.2f}, P75={desc['75%']:.2f}, max={desc['max']:.2f}")
        logger.info(f"\nCompare to current flat estimates:")
        logger.info(f"  h2h_h1_home_avg (current): 2.785 (constant)")
        logger.info(f"  h2h_h1_away_avg (current): 3.635 (constant)")
    else:
        logger.warning("No H1 odds fetched!")


if __name__ == "__main__":
    main()
