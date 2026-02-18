#!/usr/bin/env python3
"""
Collect lineups data for leagues/seasons missing lineups.parquet.

Creates lineups.parquet with the same schema as existing files:
  fixture_id, date, status, home_team, away_team, score_home, score_away,
  team_name, type, formation, id, name, number, pos, grid, coach_name, coach_id

Reads fixture IDs from matches.parquet and calls API-Football /fixtures/lineups.

Resumable: skips fixtures already in lineups.parquet.

Usage:
    python scripts/collect_lineups.py --league turkish_super_lig --season 2023 2024 2025
    python scripts/collect_lineups.py --league belgian_pro_league
    python scripts/collect_lineups.py --league belgian_pro_league --max-calls 2000
    python scripts/collect_lineups.py --dry-run --league belgian_pro_league
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "01-raw"

API_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_BASE = os.environ.get("API_BASE_URL", "https://v3.football.api-sports.io")
PER_MIN_LIMIT = int(os.environ.get("PER_MIN_LIMIT", "300"))


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


def flatten_lineups(
    api_response: list[dict], match_info: dict
) -> list[dict]:
    """Flatten API lineups response into rows matching existing schema."""
    rows = []
    for team_data in api_response:
        team_name = team_data.get("team", {}).get("name", "")
        formation = team_data.get("formation", "")
        coach = team_data.get("coach", {})
        coach_name = coach.get("name") if isinstance(coach, dict) else None
        coach_id = coach.get("id") if isinstance(coach, dict) else None

        # StartXI players
        for player_entry in team_data.get("startXI", []):
            p = player_entry.get("player", {})
            rows.append({
                "fixture_id": match_info["fixture_id"],
                "date": match_info["date"],
                "status": match_info["status"],
                "home_team": match_info["home_team"],
                "away_team": match_info["away_team"],
                "score_home": match_info["score_home"],
                "score_away": match_info["score_away"],
                "team_name": team_name,
                "type": "StartXI",
                "formation": formation,
                "id": p.get("id"),
                "name": p.get("name", ""),
                "number": p.get("number"),
                "pos": p.get("pos", ""),
                "grid": p.get("grid", ""),
                "coach_name": coach_name,
                "coach_id": coach_id,
            })

        # Substitutes
        for player_entry in team_data.get("substitutes", []):
            p = player_entry.get("player", {})
            rows.append({
                "fixture_id": match_info["fixture_id"],
                "date": match_info["date"],
                "status": match_info["status"],
                "home_team": match_info["home_team"],
                "away_team": match_info["away_team"],
                "score_home": match_info["score_home"],
                "score_away": match_info["score_away"],
                "team_name": team_name,
                "type": "Sub",
                "formation": formation,
                "id": p.get("id"),
                "name": p.get("name", ""),
                "number": p.get("number"),
                "pos": p.get("pos", ""),
                "grid": None,
                "coach_name": coach_name,
                "coach_id": coach_id,
            })

    return rows


def get_match_info(matches_df: pd.DataFrame, fixture_id: int) -> dict:
    """Extract match info from matches dataframe for a fixture."""
    row = matches_df[matches_df["fixture.id"] == fixture_id]
    if row.empty:
        return {}
    row = row.iloc[0]
    return {
        "fixture_id": int(fixture_id),
        "date": row.get("fixture.date", ""),
        "status": row.get("fixture.status.short", "FT"),
        "home_team": row.get("teams.home.name", ""),
        "away_team": row.get("teams.away.name", ""),
        "score_home": row.get("goals.home", 0),
        "score_away": row.get("goals.away", 0),
    }


def collect_season_lineups(
    league: str,
    season: str,
    matches_df: pd.DataFrame,
    max_calls: int | None,
    total_calls: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Collect lineups for a single season. Returns (calls_made, rows_created)."""
    lineups_path = RAW_DIR / league / season / "lineups.parquet"

    # Get completed fixture IDs from matches
    completed = matches_df[matches_df["fixture.status.short"] == "FT"]
    all_fixture_ids = set(completed["fixture.id"].astype(int).tolist())

    # Check which fixtures already have lineups
    existing_ids = set()
    if lineups_path.exists():
        existing_df = pd.read_parquet(lineups_path)
        existing_ids = set(existing_df["fixture_id"].unique().astype(int))

    needed = sorted(all_fixture_ids - existing_ids)
    if not needed:
        logger.info(f"{league}/{season}: all {len(all_fixture_ids)} fixtures have lineups")
        return 0, 0

    logger.info(f"{league}/{season}: {len(needed)} fixtures need lineups (of {len(all_fixture_ids)} total)")

    if dry_run:
        return 0, 0

    all_rows = []
    calls_made = 0

    for i, fixture_id in enumerate(needed):
        if max_calls and (total_calls + calls_made) >= max_calls:
            logger.warning(f"Reached max_calls limit ({max_calls}), stopping")
            break

        try:
            response = fetch_lineups(fixture_id)
            calls_made += 1

            if response:
                match_info = get_match_info(matches_df, fixture_id)
                if match_info:
                    rows = flatten_lineups(response, match_info)
                    all_rows.extend(rows)

            # Rate limiting
            if (total_calls + calls_made) % PER_MIN_LIMIT == 0:
                logger.info(
                    f"  [{league}/{season}] {total_calls + calls_made} total calls, "
                    f"{len(all_rows)} rows collected. Sleeping 62s..."
                )
                time.sleep(62)

            # Progress
            if (i + 1) % 50 == 0:
                logger.info(f"  [{league}/{season}] {i + 1}/{len(needed)} fixtures")

        except Exception as e:
            logger.error(f"Error for fixture {fixture_id}: {e}")
            time.sleep(5)

    # Save results
    if all_rows:
        new_df = pd.DataFrame(all_rows)

        if lineups_path.exists():
            existing_df = pd.read_parquet(lineups_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_parquet(lineups_path, index=False)
        logger.info(
            f"  [{league}/{season}] Saved {len(all_rows)} new rows "
            f"({len(combined)} total) to {lineups_path}"
        )

    return calls_made, len(all_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Collect lineups data from API-Football."
    )
    parser.add_argument(
        "--league", nargs="+", required=True,
        help="League(s) to collect lineups for",
    )
    parser.add_argument(
        "--season", nargs="*", default=None,
        help="Specific season(s) (default: all with matches.parquet)",
    )
    parser.add_argument(
        "--max-calls", type=int, default=None,
        help="Maximum total API calls",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be collected without API calls",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload modified lineups files to HF Hub after collection",
    )
    args = parser.parse_args()

    if not API_KEY and not args.dry_run:
        logger.error("API_FOOTBALL_KEY not set")
        return

    total_calls = 0
    total_rows = 0
    modified_files: list[Path] = []

    for league in args.league:
        logger.info(f"=== Collecting lineups for {league} ===")
        league_dir = RAW_DIR / league

        if not league_dir.exists():
            logger.error(f"League directory not found: {league_dir}")
            continue

        seasons = sorted(
            s.name for s in league_dir.iterdir()
            if s.is_dir() and s.name.isdigit()
        )
        if args.season:
            seasons = [s for s in seasons if s in args.season]

        for season in seasons:
            matches_path = league_dir / season / "matches.parquet"
            if not matches_path.exists():
                logger.info(f"{league}/{season}: no matches.parquet, skipping")
                continue

            matches_df = pd.read_parquet(matches_path)
            calls, rows = collect_season_lineups(
                league, season, matches_df,
                max_calls=args.max_calls,
                total_calls=total_calls,
                dry_run=args.dry_run,
            )
            total_calls += calls
            total_rows += rows

            if rows > 0:
                lineups_file = league_dir / season / "lineups.parquet"
                if lineups_file.exists():
                    modified_files.append(lineups_file.resolve())

            if args.max_calls and total_calls >= args.max_calls:
                logger.warning("Max calls reached, stopping all collection")
                break

        logger.info(f"Result for {league}: {total_calls} calls, {total_rows} rows")

    logger.info(f"\n=== TOTAL: {total_calls} API calls, {total_rows} rows created ===")

    if args.upload and modified_files:
        from scripts.collect_all_stats import upload_files_to_hf

        logger.info(f"Uploading {len(modified_files)} modified files to HF Hub...")
        uploaded = upload_files_to_hf(modified_files)
        logger.info(f"Uploaded {uploaded} files")
    elif args.upload:
        logger.info("No files modified, nothing to upload")


if __name__ == "__main__":
    main()
