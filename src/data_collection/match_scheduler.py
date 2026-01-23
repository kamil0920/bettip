"""
Match Schedule Manager

Fetches daily match schedule and determines when to trigger lineup collection.
Optimizes API calls by only fetching data when matches are imminent.

Usage:
    # Morning: Fetch and save today's schedule
    python -m src.data_collection.match_scheduler --fetch

    # Every 15 mins: Check if any match needs lineup collection
    python -m src.data_collection.match_scheduler --check
"""
import argparse
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Schedule file location
SCHEDULE_FILE = Path("data/06-prematch/today_schedule.json")
INTERESTING_FILE = Path("data/06-prematch/interesting_matches.json")

# Default edge threshold for filtering
DEFAULT_EDGE_THRESHOLD = 0.05

# Leagues to monitor
LEAGUE_IDS = {
    'premier_league': 39,
    'la_liga': 140,
    'serie_a': 135,
    'bundesliga': 78,
    'ligue_1': 61,
}


class MatchScheduleManager:
    """
    Manages match schedules to optimize API calls.

    Instead of polling every 30 mins, we:
    1. Fetch schedule once per day (morning)
    2. Check local schedule every 15 mins
    3. Only call API when match is 40-50 mins away
    """

    def __init__(self, schedule_file: Path = SCHEDULE_FILE):
        self.schedule_file = schedule_file
        self.schedule_file.parent.mkdir(parents=True, exist_ok=True)

    def fetch_daily_schedule(
        self,
        leagues: List[str] = None,
        days_ahead: int = 1
    ) -> Dict[str, Any]:
        """
        Fetch match schedule for today and tomorrow.

        Args:
            leagues: List of league keys to fetch
            days_ahead: Days to look ahead (default 1)

        Returns:
            Schedule dict with matches and metadata
        """
        from src.data_collection.api_client import FootballAPIClient

        if leagues is None:
            leagues = list(LEAGUE_IDS.keys())

        client = FootballAPIClient()
        all_matches = []

        today = datetime.now(timezone.utc).date()

        for league in leagues:
            league_id = LEAGUE_IDS.get(league)
            if not league_id:
                continue

            # Fetch fixtures for date range
            for day_offset in range(days_ahead + 1):
                date = today + timedelta(days=day_offset)
                date_str = date.strftime("%Y-%m-%d")

                try:
                    # Use current season (2025 for 2025-26 season)
                    current_season = 2025
                    response = client._make_request(
                        "/fixtures",
                        {
                            "league": league_id,
                            "season": current_season,
                            "date": date_str,
                        }
                    )

                    fixtures = response.get("response", [])

                    for fix in fixtures:
                        fixture = fix.get("fixture", {})
                        teams = fix.get("teams", {})

                        # Parse kickoff time
                        kickoff_str = fixture.get("date")
                        if kickoff_str:
                            kickoff = datetime.fromisoformat(
                                kickoff_str.replace("Z", "+00:00")
                            )
                        else:
                            continue

                        match = {
                            "fixture_id": fixture.get("id"),
                            "league": league,
                            "home_team": teams.get("home", {}).get("name"),
                            "away_team": teams.get("away", {}).get("name"),
                            "home_team_id": teams.get("home", {}).get("id"),
                            "away_team_id": teams.get("away", {}).get("id"),
                            "kickoff": kickoff.isoformat(),
                            "kickoff_unix": int(kickoff.timestamp()),
                            "venue": fixture.get("venue", {}).get("name"),
                            "status": fixture.get("status", {}).get("short"),
                        }

                        # Only include scheduled matches (not finished/live)
                        if match["status"] in ["NS", "TBD", None]:
                            all_matches.append(match)

                    logger.info(f"Fetched {len(fixtures)} fixtures for {league} on {date_str}")

                except Exception as e:
                    logger.error(f"Error fetching {league} fixtures: {e}")

        # Sort by kickoff time
        all_matches.sort(key=lambda x: x["kickoff_unix"])

        schedule = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "date": today.isoformat(),
            "matches": all_matches,
            "total_matches": len(all_matches),
        }

        # Save to file
        with open(self.schedule_file, "w") as f:
            json.dump(schedule, f, indent=2)

        logger.info(f"Saved schedule with {len(all_matches)} matches to {self.schedule_file}")

        return schedule

    def load_schedule(self) -> Optional[Dict[str, Any]]:
        """Load schedule from file."""
        if not self.schedule_file.exists():
            return None

        try:
            with open(self.schedule_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading schedule: {e}")
            return None

    def get_matches_in_window(
        self,
        min_minutes: int = 40,
        max_minutes: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get matches starting within the specified time window.

        Args:
            min_minutes: Minimum minutes until kickoff
            max_minutes: Maximum minutes until kickoff

        Returns:
            List of matches in the window
        """
        schedule = self.load_schedule()
        if not schedule:
            logger.warning("No schedule found. Run --fetch first.")
            return []

        now = datetime.now(timezone.utc)
        min_time = now + timedelta(minutes=min_minutes)
        max_time = now + timedelta(minutes=max_minutes)

        matches_in_window = []

        for match in schedule.get("matches", []):
            kickoff = datetime.fromisoformat(match["kickoff"])

            if min_time <= kickoff <= max_time:
                mins_until = int((kickoff - now).total_seconds() / 60)
                match["mins_until_kickoff"] = mins_until
                matches_in_window.append(match)

        return matches_in_window

    def get_next_collection_time(self) -> Optional[datetime]:
        """
        Get the next time we should collect lineup data.

        Returns time of (next_kickoff - 45 mins) or None if no matches.
        """
        schedule = self.load_schedule()
        if not schedule:
            return None

        now = datetime.now(timezone.utc)

        for match in schedule.get("matches", []):
            kickoff = datetime.fromisoformat(match["kickoff"])
            collection_time = kickoff - timedelta(minutes=45)

            # If collection time is in the future, return it
            if collection_time > now:
                return collection_time

        return None

    def should_collect_now(
        self,
        window_minutes: int = 5
    ) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Check if we should collect lineup data now.

        Returns True if any match kickoff is 40-50 minutes away.
        This gives a 10-minute window to catch the 45-min mark.

        Args:
            window_minutes: Size of the check window (default 5 mins each side)

        Returns:
            Tuple of (should_collect, list_of_matches)
        """
        matches = self.get_matches_in_window(
            min_minutes=45 - window_minutes,
            max_minutes=45 + window_minutes
        )

        return len(matches) > 0, matches

    def get_interesting_matches_in_window(
        self,
        min_minutes: int = 40,
        max_minutes: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get only INTERESTING matches in the lineup window.

        Filters matches that were marked as interesting during early prediction.
        """
        # Get all matches in window
        all_matches = self.get_matches_in_window(min_minutes, max_minutes)

        # Load interesting matches
        if not INTERESTING_FILE.exists():
            logger.warning("No interesting matches file. Using all matches.")
            return all_matches

        try:
            with open(INTERESTING_FILE) as f:
                interesting_data = json.load(f)
            interesting_ids = {m["fixture_id"] for m in interesting_data.get("matches", [])}
        except Exception as e:
            logger.error(f"Error loading interesting matches: {e}")
            return all_matches

        # Filter to only interesting matches
        filtered = [m for m in all_matches if m["fixture_id"] in interesting_ids]
        logger.info(f"Filtered {len(all_matches)} matches to {len(filtered)} interesting ones")

        return filtered


def generate_early_predictions(
    matches: List[Dict[str, Any]],
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Generate early predictions for matches and filter interesting ones.

    Runs model predictions without lineup data to identify matches
    with potential betting value (edge > threshold).

    Args:
        matches: List of match dicts from schedule
        edge_threshold: Minimum edge to consider interesting (default 0.05 = 5%)

    Returns:
        Dict with interesting matches and their predictions
    """
    from src.features.engineers.prematch import create_prematch_features_for_fixture
    from src.data_collection.prematch_collector import PreMatchCollector

    collector = PreMatchCollector()
    interesting_matches = []
    all_predictions = []

    logger.info(f"Generating early predictions for {len(matches)} matches...")

    for match in matches:
        fixture_id = match["fixture_id"]
        home_team = match["home_team"]
        away_team = match["away_team"]

        try:
            # Collect basic pre-match data (no lineups yet)
            prematch_data = collector.collect_prematch_data(fixture_id)

            # Get predictions from API-Football (bookmaker consensus)
            predictions = prematch_data.get("predictions", {})
            winner = predictions.get("winner", {})

            # Extract implied probabilities from predictions
            home_prob = None
            away_prob = None
            draw_prob = None

            if winner:
                # API-Football provides winner prediction with confidence
                winner_name = winner.get("name")
                winner_comment = (winner.get("comment") or "").lower()

                # Parse confidence from comment (e.g., "Win or draw" vs "Clear win")
                if "clear" in winner_comment:
                    confidence = 0.70
                elif "win or draw" in winner_comment:
                    confidence = 0.50
                else:
                    confidence = 0.55

                if winner_name and winner_name == home_team:
                    home_prob = confidence
                    away_prob = 0.20
                elif winner_name and winner_name == away_team:
                    away_prob = confidence
                    home_prob = 0.20
                else:
                    home_prob = 0.35
                    away_prob = 0.35

            # Get percent predictions (always available)
            percent = predictions.get("percent", {})
            home_pct_str = percent.get("home", "0%")
            away_pct_str = percent.get("away", "0%")
            draw_pct_str = percent.get("draw", "0%")

            # Parse percentages
            home_pct = float(home_pct_str.replace("%", "")) / 100 if home_pct_str else 0
            away_pct = float(away_pct_str.replace("%", "")) / 100 if away_pct_str else 0
            draw_pct = float(draw_pct_str.replace("%", "")) / 100 if draw_pct_str else 0

            # Get odds if available
            odds = prematch_data.get("odds", {})
            h2h = odds.get("h2h", {})
            home_odds = h2h.get("home") if h2h else None
            away_odds = h2h.get("away") if h2h else None

            # Calculate edges
            edges = {}
            if home_odds and away_odds:
                # If odds available, calculate edge vs bookmaker
                total_implied = 1/home_odds + 1/away_odds + (1/h2h.get("draw", 3.5) if h2h.get("draw") else 0.28)
                home_implied = (1/home_odds) / total_implied
                away_implied = (1/away_odds) / total_implied

                if home_pct:
                    edges["home_win"] = home_pct - home_implied
                if away_pct:
                    edges["away_win"] = away_pct - away_implied
            else:
                # No odds - use prediction strength as "edge"
                # If API strongly predicts one side (>55%), mark as interesting
                # Edge = how far above baseline (33%) the prediction is
                baseline = 0.33
                if home_pct > baseline:
                    edges["home_win"] = home_pct - baseline
                if away_pct > baseline:
                    edges["away_win"] = away_pct - baseline

            # Check BTTS signal
            goals = predictions.get("goals", {})
            if goals:
                btts_signal = str(goals.get("home", "")).startswith("-") and str(goals.get("away", "")).startswith("-")
                if btts_signal:
                    edges["under_2.5"] = 0.10  # Flag under as potentially interesting

            max_edge = max(edges.values()) if edges else 0
            best_market = max(edges, key=edges.get) if edges else None

            prediction = {
                "fixture_id": fixture_id,
                "match": f"{home_team} vs {away_team}",
                "kickoff": match["kickoff"],
                "league": match["league"],
                "home_pct": home_pct,
                "away_pct": away_pct,
                "draw_pct": draw_pct,
                "home_odds": home_odds,
                "away_odds": away_odds,
                "edges": edges,
                "max_edge": max_edge,
                "best_market": best_market,
                "is_interesting": max_edge >= edge_threshold,
                "api_advice": predictions.get("advice"),
            }

            all_predictions.append(prediction)

            if max_edge >= edge_threshold:
                interesting_matches.append({
                    **match,
                    "prediction": prediction,
                })
                logger.info(
                    f"  ✓ {home_team} vs {away_team}: "
                    f"{home_pct:.0%}/{draw_pct:.0%}/{away_pct:.0%} - "
                    f"edge={max_edge:.1%} on {best_market}"
                )
            else:
                logger.debug(
                    f"  ✗ {home_team} vs {away_team}: "
                    f"edge={max_edge:.1%} (below threshold)"
                )

        except Exception as e:
            logger.warning(f"Error predicting {home_team} vs {away_team}: {e}")

    # Save interesting matches
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_matches": len(matches),
        "interesting_count": len(interesting_matches),
        "edge_threshold": edge_threshold,
        "matches": interesting_matches,
        "all_predictions": all_predictions,
    }

    INTERESTING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INTERESTING_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(
        f"Found {len(interesting_matches)}/{len(matches)} interesting matches "
        f"(edge >= {edge_threshold:.0%})"
    )

    return result


def fetch_match_weather(venue: str, kickoff: datetime) -> Optional[Dict[str, Any]]:
    """
    Fetch weather forecast for match venue at kickoff time.

    Uses Open-Meteo forecast API (not archive).

    Args:
        venue: Venue city name
        kickoff: Match kickoff datetime

    Returns:
        Weather dict or None
    """
    import requests
    from src.data_collection.weather_collector import CITY_COORDINATES

    # Get coordinates
    coords = CITY_COORDINATES.get(venue)
    if not coords:
        # Try partial match
        for city, city_coords in CITY_COORDINATES.items():
            if city.lower() in venue.lower() or venue.lower() in city.lower():
                coords = city_coords
                break

    if not coords:
        return None

    lat, lon = coords

    try:
        # Use forecast API (not archive)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "UTC",
            "forecast_days": 2,
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            hourly = data.get("hourly", {})

            if hourly and hourly.get("time"):
                # Find the hour closest to kickoff
                times = hourly["time"]
                kickoff_hour = kickoff.strftime("%Y-%m-%dT%H:00")

                idx = None
                for i, t in enumerate(times):
                    if t == kickoff_hour:
                        idx = i
                        break

                if idx is not None:
                    temp = hourly["temperature_2m"][idx]
                    precip = hourly["precipitation"][idx]
                    wind = hourly["wind_speed_10m"][idx]
                    code = hourly["weather_code"][idx]

                    return {
                        "temperature": temp,
                        "precipitation": precip,
                        "wind_speed": wind,
                        "weather_code": code,
                        "is_rainy": precip > 0.5,
                        "is_windy": wind > 25,
                        "is_extreme": temp < 5 or temp > 30,
                        "conditions": _weather_code_to_text(code),
                    }

    except Exception as e:
        logger.warning(f"Weather fetch failed for {venue}: {e}")

    return None


def _weather_code_to_text(code: int) -> str:
    """Convert WMO weather code to human text."""
    if code <= 3:
        return "Clear/Cloudy"
    elif 45 <= code <= 48:
        return "Foggy"
    elif 51 <= code <= 57:
        return "Drizzle"
    elif 61 <= code <= 67:
        return "Rain"
    elif 71 <= code <= 77:
        return "Snow"
    elif 80 <= code <= 82:
        return "Rain showers"
    elif 95 <= code <= 99:
        return "Thunderstorm"
    return "Unknown"


def collect_lineups_for_matches(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collect lineup and weather data for specified matches.

    Args:
        matches: List of match dicts with fixture_id

    Returns:
        Collection results
    """
    from src.data_collection.prematch_collector import PreMatchCollector
    from src.ml.confidence_adjuster import LineupConfidenceAdjuster

    collector = PreMatchCollector()
    adjuster = LineupConfidenceAdjuster()

    results = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "matches_checked": len(matches),
        "lineups_found": 0,
        "weather_collected": 0,
        "recommendations": [],
    }

    for match in matches:
        fixture_id = match["fixture_id"]
        logger.info(f"Collecting data for {match['home_team']} vs {match['away_team']}")

        try:
            # Collect pre-match data
            prematch = collector.collect_prematch_data(fixture_id)

            lineups = prematch.get("lineups", {})

            if not lineups.get("available"):
                logger.info(f"  Lineups not yet available for fixture {fixture_id}")
                continue

            results["lineups_found"] += 1
            logger.info(f"  ✓ Lineups available!")

            # Analyze lineups
            home_analysis = adjuster.analyze_lineup(
                match["home_team_id"],
                match["home_team"],
                lineups.get("home", {})
            )
            away_analysis = adjuster.analyze_lineup(
                match["away_team_id"],
                match["away_team"],
                lineups.get("away", {})
            )

            # Fetch weather forecast
            venue = match.get("venue", "")
            kickoff_dt = datetime.fromisoformat(match["kickoff"])
            weather = fetch_match_weather(venue, kickoff_dt)

            if weather:
                results["weather_collected"] += 1
                logger.info(f"  ✓ Weather: {weather['conditions']}, {weather['temperature']:.0f}°C")

            # Build recommendation
            rec = {
                "fixture_id": fixture_id,
                "match": f"{match['home_team']} vs {match['away_team']}",
                "kickoff": match["kickoff"],
                "mins_until": match.get("mins_until_kickoff", 45),
                "venue": venue,
                "home_formation": lineups.get("home", {}).get("formation"),
                "away_formation": lineups.get("away", {}).get("formation"),
                "home_strength": home_analysis.strength_score,
                "away_strength": away_analysis.strength_score,
                "home_missing": home_analysis.key_players_missing,
                "away_missing": away_analysis.key_players_missing,
                "weather": weather,
                "signals": [],
            }

            # Generate signals based on key player absences
            if len(home_analysis.key_players_missing) >= 2:
                rec["signals"].append({
                    "market": "Away +0.5 AH",
                    "confidence": 0.60,
                    "reason": f"Home missing: {', '.join(home_analysis.key_players_missing[:2])}"
                })

            if len(away_analysis.key_players_missing) >= 2:
                rec["signals"].append({
                    "market": "Home -0.5 AH",
                    "confidence": 0.60,
                    "reason": f"Away missing: {', '.join(away_analysis.key_players_missing[:2])}"
                })

            # Strength imbalance signals
            if home_analysis.strength_score >= 0.9 and away_analysis.strength_score < 0.7:
                rec["signals"].append({
                    "market": "Home Win",
                    "confidence": 0.58,
                    "reason": f"Home full strength vs weakened away ({away_analysis.strength_score:.0%})"
                })
            elif away_analysis.strength_score >= 0.9 and home_analysis.strength_score < 0.7:
                rec["signals"].append({
                    "market": "Away Win",
                    "confidence": 0.55,
                    "reason": f"Away full strength vs weakened home ({home_analysis.strength_score:.0%})"
                })

            # Weather-based signals
            if weather:
                # Heavy rain favors home team (away travel, adaptation)
                if weather.get("is_rainy") and weather.get("precipitation", 0) > 2:
                    rec["signals"].append({
                        "market": "Home +0.5 AH",
                        "confidence": 0.55,
                        "reason": f"Heavy rain ({weather['precipitation']:.1f}mm) - favors home team"
                    })
                    # Rain often means fewer goals
                    rec["signals"].append({
                        "market": "Under 2.5",
                        "confidence": 0.55,
                        "reason": f"Rainy conditions reduce scoring"
                    })

                # Strong wind disrupts play, favors defensive teams
                if weather.get("is_windy") and weather.get("wind_speed", 0) > 30:
                    rec["signals"].append({
                        "market": "Under 2.5",
                        "confidence": 0.57,
                        "reason": f"Strong wind ({weather['wind_speed']:.0f} km/h) disrupts play"
                    })

                # Extreme cold affects player performance
                if weather.get("temperature", 15) < 3:
                    rec["signals"].append({
                        "market": "Under 2.5",
                        "confidence": 0.55,
                        "reason": f"Extreme cold ({weather['temperature']:.0f}°C) reduces intensity"
                    })

            results["recommendations"].append(rec)

        except Exception as e:
            logger.error(f"Error collecting data for fixture {fixture_id}: {e}")

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Match Schedule Manager")
    parser.add_argument("--fetch", action="store_true",
                       help="Fetch today's match schedule")
    parser.add_argument("--predict", action="store_true",
                       help="Run early predictions to filter interesting matches")
    parser.add_argument("--check", action="store_true",
                       help="Check if lineup collection needed now")
    parser.add_argument("--collect", action="store_true",
                       help="Collect lineups for interesting matches in window")
    parser.add_argument("--all", action="store_true",
                       help="With --collect: collect ALL matches, not just interesting")
    parser.add_argument("--next", action="store_true",
                       help="Show next collection time")
    parser.add_argument("--leagues", nargs="+", default=None,
                       help="Leagues to fetch (default: all)")
    parser.add_argument("--edge", type=float, default=DEFAULT_EDGE_THRESHOLD,
                       help=f"Edge threshold for interesting matches (default: {DEFAULT_EDGE_THRESHOLD})")

    args = parser.parse_args()

    manager = MatchScheduleManager()

    if args.fetch:
        schedule = manager.fetch_daily_schedule(leagues=args.leagues)
        print(f"\nFetched {schedule['total_matches']} matches:")
        for match in schedule["matches"][:10]:
            kickoff = datetime.fromisoformat(match["kickoff"])
            print(f"  {kickoff.strftime('%H:%M')} - {match['home_team']} vs {match['away_team']}")
        if schedule["total_matches"] > 10:
            print(f"  ... and {schedule['total_matches'] - 10} more")

    if args.predict:
        schedule = manager.load_schedule()
        if not schedule:
            print("No schedule found. Run --fetch first.")
        else:
            matches = schedule.get("matches", [])
            print(f"\nRunning early predictions for {len(matches)} matches...")
            result = generate_early_predictions(matches, edge_threshold=args.edge)

            print(f"\n{'='*50}")
            print(f"EARLY PREDICTION RESULTS")
            print(f"{'='*50}")
            print(f"Total matches: {result['total_matches']}")
            print(f"Interesting matches: {result['interesting_count']}")
            print(f"Edge threshold: {result['edge_threshold']:.0%}")
            print(f"\nInteresting matches saved to: {INTERESTING_FILE}")

            if result["matches"]:
                print(f"\n📊 Interesting matches:")
                for m in result["matches"]:
                    pred = m.get("prediction", {})
                    print(f"  • {m['home_team']} vs {m['away_team']}")
                    print(f"    Edge: {pred.get('max_edge', 0):.1%} on {pred.get('best_market', 'N/A')}")

    if args.check:
        # Check for interesting matches only
        if args.all:
            should_collect, matches = manager.should_collect_now()
        else:
            matches = manager.get_interesting_matches_in_window()
            should_collect = len(matches) > 0

        if should_collect:
            print(f"\n✓ COLLECT NOW! {len(matches)} {'interesting ' if not args.all else ''}match(es) in lineup window:")
            for m in matches:
                print(f"  {m['home_team']} vs {m['away_team']} "
                      f"(kickoff in {m.get('mins_until_kickoff', '?')} mins)")
        else:
            print("\n✗ No interesting matches in lineup window right now")
            next_time = manager.get_next_collection_time()
            if next_time:
                mins_until = int((next_time - datetime.now(timezone.utc)).total_seconds() / 60)
                print(f"  Next collection at {next_time.strftime('%H:%M UTC')} ({mins_until} mins)")

    if args.collect:
        # Collect only interesting matches (unless --all is specified)
        if args.all:
            should_collect, matches = manager.should_collect_now()
        else:
            matches = manager.get_interesting_matches_in_window()
            should_collect = len(matches) > 0

        if should_collect:
            mode = "ALL" if args.all else "INTERESTING"
            print(f"\nCollecting lineup data for {len(matches)} {mode} matches...")
            results = collect_lineups_for_matches(matches)

            print(f"\nResults:")
            print(f"  Matches checked: {results['matches_checked']}")
            print(f"  Lineups found: {results['lineups_found']}")
            print(f"  Recommendations: {len(results['recommendations'])}")

            # Save results
            output_file = Path("data/06-prematch/lineup_collection.json")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nSaved to {output_file}")
        else:
            print("No interesting matches in lineup window")
            if not args.all:
                print("  (Use --all to collect ALL matches regardless of predictions)")

    if args.next:
        next_time = manager.get_next_collection_time()
        if next_time:
            mins_until = int((next_time - datetime.now(timezone.utc)).total_seconds() / 60)
            print(f"Next lineup collection: {next_time.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"  ({mins_until} minutes from now)")
        else:
            print("No upcoming matches in schedule")


if __name__ == "__main__":
    main()
