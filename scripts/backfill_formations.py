#!/usr/bin/env python3
"""
Backfill formation data into existing lineups.parquet files.

Derives formation from the grid column (e.g., "2:3" -> row 2, col 3) already
present in StartXI rows. Counts players per grid row (skipping row 1 = GK)
to construct formation strings like "4-3-3". Zero API calls needed.

Usage:
    python scripts/backfill_formations.py                    # All leagues/seasons
    python scripts/backfill_formations.py --league la_liga   # Single league
    python scripts/backfill_formations.py --dry-run          # Preview only
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "01-raw"


def derive_formation(grid_values: pd.Series) -> str | None:
    """Derive formation string from grid positions of StartXI players.

    Grid format is "row:col" where row 1 = GK. We count outfield players
    per row (rows 2+) and join counts with '-'.
    E.g., grids with rows [1,2,2,2,2,3,3,3,4,4,4] -> "4-3-3"
    """
    grids = grid_values.dropna()
    if len(grids) < 10:  # need at least 10 outfield players
        return None

    rows = []
    for g in grids:
        try:
            row = int(str(g).split(":")[0])
            if row > 1:  # skip GK
                rows.append(row)
        except (ValueError, IndexError):
            continue

    if not rows:
        return None

    from collections import Counter
    counts = Counter(rows)
    # Sort by row number, join counts
    formation = "-".join(str(counts[r]) for r in sorted(counts.keys()))
    return formation if formation else None


def backfill_file(parquet_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Backfill formation for one lineups.parquet file.

    Returns (fixtures_processed, fixtures_filled).
    """
    df = pd.read_parquet(parquet_path)

    if "formation" in df.columns and df["formation"].notna().all():
        logger.info(f"  SKIP {parquet_path.relative_to(RAW_DIR)} - all formations present")
        return 0, 0

    if "formation" not in df.columns:
        df["formation"] = None

    if "grid" not in df.columns:
        logger.warning(f"  SKIP {parquet_path.relative_to(RAW_DIR)} - no grid column")
        return 0, 0

    # Only derive for rows missing formation
    missing_mask = df["formation"].isna()
    missing_fixtures = df.loc[missing_mask, "fixture_id"].unique()

    if len(missing_fixtures) == 0:
        logger.info(f"  SKIP {parquet_path.relative_to(RAW_DIR)} - no missing formations")
        return 0, 0

    if dry_run:
        logger.info(f"  {parquet_path.relative_to(RAW_DIR)}: {len(missing_fixtures)} fixtures to fill")
        return len(missing_fixtures), 0

    filled = 0
    startxi = df[df["type"] == "StartXI"]

    for fix_id in missing_fixtures:
        fix_rows = startxi[startxi["fixture_id"] == fix_id]
        for team_name, team_rows in fix_rows.groupby("team_name"):
            formation = derive_formation(team_rows["grid"])
            if formation:
                mask = (df["fixture_id"] == fix_id) & (df["team_name"] == team_name)
                df.loc[mask, "formation"] = formation
                filled += 1

    df.to_parquet(parquet_path, index=False)
    rel = parquet_path.relative_to(RAW_DIR)
    logger.info(f"  {rel}: {filled} team-formations derived from grid for {len(missing_fixtures)} fixtures")
    return len(missing_fixtures), filled


def main():
    parser = argparse.ArgumentParser(description="Backfill formation from grid column")
    parser.add_argument("--league", help="Single league to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    files = sorted(RAW_DIR.rglob("lineups.parquet"))
    if args.league:
        files = [f for f in files if args.league in str(f)]

    total_fixtures = 0
    total_filled = 0
    for f in files:
        fixtures, filled = backfill_file(f, dry_run=args.dry_run)
        total_fixtures += fixtures
        total_filled += filled

    if args.dry_run:
        logger.info(f"\nDry run: {total_fixtures} fixtures need formation backfill (0 API calls)")
    else:
        logger.info(f"\nDone: {total_filled} team-formations derived across {total_fixtures} fixtures")


if __name__ == "__main__":
    main()
