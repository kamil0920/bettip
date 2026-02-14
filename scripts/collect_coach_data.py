#!/usr/bin/env python3
"""
Collect coach data for leagues missing it in lineups.parquet files.

For each fixture without coach data, calls the API-Football /fixtures/lineups
endpoint and adds coach_name + coach_id to existing lineups rows.

Resumable: skips fixtures that already have coach data.

Usage:
    python scripts/collect_coach_data.py --league serie_a la_liga
    python scripts/collect_coach_data.py --league serie_a --season 2024 2025
    python scripts/collect_coach_data.py --league serie_a --max-calls 1000
    python scripts/collect_coach_data.py --dry-run --league serie_a
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "01-raw"

API_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_BASE = os.environ.get("API_BASE_URL", "https://v3.football.api-sports.io")
PER_MIN_LIMIT = int(os.environ.get("PER_MIN_LIMIT", "30"))


def fetch_lineups(fixture_id: int) -> list[dict]:
    """Fetch lineups for a single fixture from API-Football."""
    headers = {"x-apisports-key": API_KEY}
    url = f"{API_BASE}/fixtures/lineups"
    params = {"fixture": fixture_id}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("errors"):
        logger.warning(f"API error for fixture {fixture_id}: {data['errors']}")
        return []

    return data.get("response", [])


def extract_coaches(api_response: list[dict]) -> dict[str, dict]:
    """Extract coach data per team from API response.

    Returns: {team_name: {'coach_name': str, 'coach_id': int}}
    """
    coaches = {}
    for team_data in api_response:
        team_name = team_data.get("team", {}).get("name")
        coach = team_data.get("coach", {})
        if team_name and isinstance(coach, dict) and coach.get("name"):
            coaches[team_name] = {
                "coach_name": coach["name"],
                "coach_id": coach.get("id"),
            }
    return coaches


def get_fixtures_needing_coach(lineups_path: Path) -> list[int]:
    """Get fixture IDs from lineups file that lack coach data."""
    if not lineups_path.exists():
        return []

    df = pd.read_parquet(lineups_path)

    # Identify the fixture_id column
    fix_col = None
    for candidate in ["fixture_id", "fixture.id"]:
        if candidate in df.columns:
            fix_col = candidate
            break

    if fix_col is None:
        logger.warning(f"No fixture_id column in {lineups_path}")
        return []

    # Check if coach data already exists
    if "coach_name" in df.columns:
        # Only get fixtures where ALL rows have NaN coach_name
        missing = df.groupby(fix_col)["coach_name"].apply(
            lambda x: x.isna().all()
        )
        return sorted(missing[missing].index.astype(int).tolist())
    else:
        # No coach columns at all — need all fixtures
        return sorted(df[fix_col].dropna().astype(int).unique().tolist())


def update_lineups_with_coaches(
    lineups_path: Path, fixture_id: int, coaches: dict[str, dict]
) -> int:
    """Update lineups parquet with coach data for a specific fixture.

    Returns: number of rows updated.
    """
    df = pd.read_parquet(lineups_path)

    fix_col = None
    for candidate in ["fixture_id", "fixture.id"]:
        if candidate in df.columns:
            fix_col = candidate
            break

    if fix_col is None:
        return 0

    # Ensure coach columns exist
    if "coach_name" not in df.columns:
        df["coach_name"] = None
    if "coach_id" not in df.columns:
        df["coach_id"] = None

    # Update rows for this fixture
    mask = df[fix_col].astype(float) == fixture_id
    updated = 0
    for idx in df[mask].index:
        team = df.at[idx, "team_name"] if "team_name" in df.columns else None
        if team and team in coaches:
            df.at[idx, "coach_name"] = coaches[team]["coach_name"]
            df.at[idx, "coach_id"] = coaches[team]["coach_id"]
            updated += 1

    if updated > 0:
        df.to_parquet(lineups_path, index=False)

    return updated


def collect_league_coaches(
    league: str,
    seasons: list[str] | None = None,
    max_calls: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Collect coach data for a league."""
    league_dir = RAW_DIR / league

    if not league_dir.exists():
        logger.error(f"League directory not found: {league_dir}")
        return {"error": "not_found"}

    available_seasons = sorted(
        [s.name for s in league_dir.iterdir() if s.is_dir() and s.name.isdigit()]
    )
    if seasons:
        available_seasons = [s for s in available_seasons if s in seasons]

    total_calls = 0
    total_updated = 0
    total_fixtures = 0

    for season in available_seasons:
        lineups_path = league_dir / season / "lineups.parquet"
        if not lineups_path.exists():
            logger.info(f"{league}/{season}: no lineups.parquet, skipping")
            continue

        fixtures = get_fixtures_needing_coach(lineups_path)
        if not fixtures:
            logger.info(f"{league}/{season}: all fixtures have coach data")
            continue

        logger.info(
            f"{league}/{season}: {len(fixtures)} fixtures need coach data"
        )
        total_fixtures += len(fixtures)

        if dry_run:
            continue

        for i, fixture_id in enumerate(fixtures):
            if max_calls and total_calls >= max_calls:
                logger.warning(
                    f"Reached max_calls limit ({max_calls}), stopping"
                )
                return {
                    "calls": total_calls,
                    "updated_rows": total_updated,
                    "fixtures_processed": total_fixtures,
                    "stopped_early": True,
                }

            try:
                response = fetch_lineups(fixture_id)
                total_calls += 1

                if response:
                    coaches = extract_coaches(response)
                    if coaches:
                        n = update_lineups_with_coaches(
                            lineups_path, fixture_id, coaches
                        )
                        total_updated += n

                # Rate limiting
                if total_calls % PER_MIN_LIMIT == 0:
                    logger.info(
                        f"  [{league}/{season}] {total_calls} calls made, "
                        f"{total_updated} rows updated. "
                        f"Sleeping 62s for rate limit..."
                    )
                    time.sleep(62)

                # Progress logging
                if (i + 1) % 50 == 0:
                    logger.info(
                        f"  [{league}/{season}] {i + 1}/{len(fixtures)} fixtures"
                    )

            except Exception as e:
                logger.error(f"Error for fixture {fixture_id}: {e}")
                time.sleep(5)  # Back off on errors

    return {
        "calls": total_calls,
        "updated_rows": total_updated,
        "fixtures_needing_update": total_fixtures,
        "stopped_early": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect coach data from API-Football for lineups files."
    )
    parser.add_argument(
        "--league",
        nargs="+",
        required=True,
        help="League(s) to collect coach data for",
    )
    parser.add_argument(
        "--season",
        nargs="*",
        default=None,
        help="Specific season(s) to collect (default: all)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Maximum API calls to make (for budget control)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be collected without making API calls",
    )
    args = parser.parse_args()

    if not API_KEY and not args.dry_run:
        logger.error("API_FOOTBALL_KEY not set in environment")
        return

    for league in args.league:
        logger.info(f"=== Collecting coach data for {league} ===")
        result = collect_league_coaches(
            league=league,
            seasons=args.season,
            max_calls=args.max_calls,
            dry_run=args.dry_run,
        )
        logger.info(f"Result for {league}: {result}")


if __name__ == "__main__":
    main()
