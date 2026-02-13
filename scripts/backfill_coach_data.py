#!/usr/bin/env python3
"""
Backfill coach data into existing lineups.parquet files.

Extracts coach_name and coach_id from the nested `lineups` JSON column already
present in raw parquet files. Zero API calls needed — data is already collected.

Two data formats are handled:
1. Files with `lineups` JSON column (e.g., Premier League) — coach data extracted
   from nested JSON. fixture_id resolved from `fixture_info` when missing.
2. Files without `lineups` column (other leagues) — skipped (coach data not
   available without API calls).

Usage:
    python scripts/backfill_coach_data.py                    # All leagues/seasons
    python scripts/backfill_coach_data.py --league premier_league  # Single league
    python scripts/backfill_coach_data.py --dry-run          # Preview only
"""
import argparse
import ast
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "01-raw"


def _parse_json(val):
    """Safely parse a string as Python literal (dict/list)."""
    if pd.isna(val) or val == "nan":
        return None
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except (ValueError, SyntaxError):
        return None


def extract_fixture_coaches(lineups_str: str, fixture_info_str=None, fixture_id=None):
    """Extract fixture_id and coach data from nested JSON columns.

    Returns (fixture_id, [{'team_name': str, 'coach_name': str, 'coach_id': int}, ...])
    """
    # Get fixture_id from fixture_info if not provided
    actual_fix_id = int(fixture_id) if pd.notna(fixture_id) else None
    if actual_fix_id is None:
        fi = _parse_json(fixture_info_str)
        if isinstance(fi, dict):
            actual_fix_id = fi.get("id")

    # Parse lineups for coach data
    lineups_val = _parse_json(lineups_str)
    if not isinstance(lineups_val, list):
        return actual_fix_id, []

    coaches = []
    for team_data in lineups_val:
        if not isinstance(team_data, dict):
            continue
        team_info = team_data.get("team", {})
        team_name = team_info.get("name") if isinstance(team_info, dict) else None
        coach_data = team_data.get("coach", {})
        if isinstance(coach_data, dict) and coach_data.get("name") and team_name:
            coaches.append({
                "team_name": team_name,
                "coach_name": coach_data["name"],
                "coach_id": coach_data.get("id"),
            })
    return actual_fix_id, coaches


def backfill_file(parquet_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Backfill coach data for one lineups.parquet file.

    Returns (fixtures_total, fixtures_filled).
    """
    df = pd.read_parquet(parquet_path)
    rel = parquet_path.relative_to(RAW_DIR)

    if "lineups" not in df.columns:
        return 0, 0

    # Check if coach data already fully populated
    if "coach_name" in df.columns and df["coach_name"].notna().all():
        logger.info(f"  SKIP {rel} - all coach data present")
        return 0, 0

    has_fixture_info = "fixture_info" in df.columns
    has_fixture_id = "fixture_id" in df.columns

    # Build lookup from unique lineups strings (one per fixture)
    # (fixture_id, team_name) -> {coach_name, coach_id}
    coaches: dict[tuple, dict] = {}
    seen = set()

    for idx in range(len(df)):
        row = df.iloc[idx]
        lu_str = row.get("lineups")
        if pd.isna(lu_str) or lu_str == "nan":
            continue

        lu_hash = id(lu_str) if not isinstance(lu_str, str) else hash(lu_str)
        if lu_hash in seen:
            continue
        seen.add(lu_hash)

        fi_str = row["fixture_info"] if has_fixture_info else None
        fix_id = row["fixture_id"] if has_fixture_id else None

        actual_fix_id, team_coaches = extract_fixture_coaches(lu_str, fi_str, fix_id)
        if actual_fix_id is None:
            continue

        for tc in team_coaches:
            coaches[(actual_fix_id, tc["team_name"])] = {
                "coach_name": tc["coach_name"],
                "coach_id": tc["coach_id"],
            }

    if not coaches:
        logger.info(f"  SKIP {rel} - no coach data found in lineups JSON")
        return 0, 0

    n_fixtures = len({k[0] for k in coaches})

    if dry_run:
        logger.info(f"  {rel}: {n_fixtures} fixtures with coach data ({len(coaches)} team entries)")
        return n_fixtures, n_fixtures

    # Step 1: Populate fixture_id from fixture_info where missing
    if has_fixture_id and df["fixture_id"].isna().any() and has_fixture_info:
        missing_fid = df["fixture_id"].isna()
        for idx in df.index[missing_fid]:
            fi = _parse_json(df.at[idx, "fixture_info"])
            if isinstance(fi, dict) and fi.get("id"):
                df.at[idx, "fixture_id"] = fi["id"]

    # Step 2: Populate team_name from lineups JSON where it's "nan" or missing
    if "team_name" in df.columns:
        bad_team = df["team_name"].isna() | (df["team_name"].astype(str) == "nan")
        if bad_team.any():
            # Build fixture_id -> (home_team, away_team) from coaches lookup
            fix_teams: dict[int, tuple[str, str]] = {}
            for (fix_id, team_name) in coaches:
                if fix_id not in fix_teams:
                    fix_teams[fix_id] = []
                fix_teams[fix_id].append(team_name)

            # For each fixture, the player rows alternate between two teams.
            # Use the 'type' column hint: StartXI players come in team order.
            # Simpler approach: assign team_name from fixture_info home/away
            if has_fixture_info:
                for idx in df.index[bad_team]:
                    fi = _parse_json(df.at[idx, "fixture_info"])
                    if not isinstance(fi, dict):
                        continue
                    fix_id = df.at[idx, "fixture_id"]
                    if pd.isna(fix_id):
                        continue
                    # Get teams for this fixture from coaches lookup
                    teams = fix_teams.get(int(fix_id), [])
                    if len(teams) == 2:
                        # Use row position within fixture to determine team
                        fix_mask = df["fixture_id"] == fix_id
                        fix_indices = df.index[fix_mask].tolist()
                        pos = fix_indices.index(idx)
                        # First half of rows = first team (home), second half = second team (away)
                        half = len(fix_indices) // 2
                        team_idx = 0 if pos < half else 1
                        df.at[idx, "team_name"] = teams[team_idx]

    # Step 3: Apply coach data
    if "coach_name" not in df.columns:
        df["coach_name"] = None
    if "coach_id" not in df.columns:
        df["coach_id"] = None

    filled = 0
    for (fix_id, team_name), coach_info in coaches.items():
        mask = (df["fixture_id"] == fix_id) & (df["team_name"] == team_name)
        if mask.any():
            df.loc[mask, "coach_name"] = coach_info["coach_name"]
            df.loc[mask, "coach_id"] = coach_info["coach_id"]
            filled += 1

    df.to_parquet(parquet_path, index=False)
    logger.info(f"  {rel}: {filled}/{len(coaches)} team-coach entries filled across {n_fixtures} fixtures")
    return n_fixtures, filled


def main():
    parser = argparse.ArgumentParser(description="Backfill coach data from nested lineups JSON")
    parser.add_argument("--league", help="Single league to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    files = sorted(RAW_DIR.rglob("lineups.parquet"))
    if args.league:
        files = [f for f in files if args.league in str(f)]

    # Quick filter: only process files with lineups column (check schema, no full read)
    eligible = []
    for f in files:
        schema = pq.read_schema(f)
        if "lineups" in schema.names:
            eligible.append(f)

    logger.info(f"Found {len(files)} lineups.parquet files, {len(eligible)} have lineups JSON column")

    total_fixtures = 0
    total_filled = 0
    for f in eligible:
        fixtures, filled = backfill_file(f, dry_run=args.dry_run)
        total_fixtures += fixtures
        total_filled += filled

    if args.dry_run:
        logger.info(f"\nDry run: {total_filled} fixtures with coach data across {len(eligible)} files (0 API calls)")
    else:
        logger.info(f"\nDone: {total_filled} team-coach entries filled across {total_fixtures} fixtures in {len(eligible)} files")


if __name__ == "__main__":
    main()
