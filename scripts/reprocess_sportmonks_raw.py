#!/usr/bin/env python3
"""Re-process raw Sportmonks CSV exports into the pivoted wide format.

The original processing script dropped 99.5% of corners data at target lines
(8.5-11.5). This script reads the raw long-format CSVs and produces clean
pivoted CSVs that sportmonks_per_line.py expects.

Raw format (one row per fixture x line x label x bookmaker):
    fixture_id, ..., label (Over/Under), line, odds, bookmaker_id

Output format (one row per fixture x line):
    fixture_id, ..., line, over_avg, over_best, over_count, under_avg, under_best, under_count

Usage:
    python scripts/reprocess_sportmonks_raw.py              # Dry-run: show stats
    python scripts/reprocess_sportmonks_raw.py --save       # Save processed CSVs
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/sportmonks_odds/raw")
OUT_DIR = Path("data/sportmonks_odds/processed")

# Target lines matching sportmonks_per_line.py TARGET_LINES
TARGET_LINES = {
    "corners": [8.5, 9.5, 10.5, 11.5],
    "cards": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
}

KEEP_COLUMNS = [
    "fixture_id",
    "fixture_name",
    "home_team",
    "away_team",
    "home_team_normalized",
    "away_team_normalized",
    "start_time",
    "league_id",
    "market_id",
    "market",
    "line",
    "over_avg",
    "over_best",
    "over_count",
    "under_avg",
    "under_best",
    "under_count",
]


def _process_corners(raw: pd.DataFrame, target_lines: list[float]) -> pd.DataFrame:
    """Process raw corners data (standard label/line columns)."""
    # Filter to Over/Under only (drop "Exactly")
    df = raw[raw["label"].isin(["Over", "Under"])].copy()

    # Filter to target lines
    df = df[df["line"].isin(target_lines)].copy()

    if df.empty:
        logger.warning("No corners rows at target lines after filtering")
        return pd.DataFrame(columns=KEEP_COLUMNS)

    logger.info(
        f"Corners: {len(df)} rows at target lines, "
        f"{df['fixture_id'].nunique()} fixtures"
    )

    # Use "Alternative Corners" as market name for all
    df["market"] = "Alternative Corners"

    return _aggregate_and_pivot(df)


def _process_cards(raw: pd.DataFrame, target_lines: list[float]) -> pd.DataFrame:
    """Process raw cards data (two market formats).

    - Asian Total Cards (market_id=272): label=Over/Under, line column has values
    - Number of Cards (market_id=255): label="3.5 | Over", line is NaN
    """
    frames = []

    # --- Format 1: Asian Total Cards (standard label/line) ---
    atc = raw[raw["market_id"] == 272].copy()
    if not atc.empty:
        atc = atc[atc["label"].isin(["Over", "Under"])]
        atc = atc[atc["line"].isin(target_lines)]
        atc["market"] = "Asian Total Cards"
        if not atc.empty:
            logger.info(f"Cards (Asian Total): {len(atc)} rows")
            frames.append(atc)

    # --- Format 2: Number of Cards (combined label, no line column) ---
    noc = raw[raw["market_id"] == 255].copy()
    if not noc.empty:
        # Parse "3.5 | Over" → line=3.5, label=Over
        parsed = noc["label"].str.extract(r"([\d.]+)\s*\|\s*(Over|Under)")
        noc = noc[parsed[0].notna()].copy()
        noc["line"] = parsed[0].astype(float)
        noc["label"] = parsed[1]
        noc = noc[noc["line"].isin(target_lines)]
        noc["market"] = "Cards Over/Under"
        if not noc.empty:
            logger.info(f"Cards (Number of Cards): {len(noc)} rows")
            frames.append(noc)

    if not frames:
        logger.warning("No cards rows at target lines after filtering")
        return pd.DataFrame(columns=KEEP_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Cards total: {len(df)} rows at target lines, "
        f"{df['fixture_id'].nunique()} fixtures"
    )

    return _aggregate_and_pivot(df)


def _aggregate_and_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per fixture x line x label, then pivot Over/Under into columns."""
    group_cols = [
        "fixture_id",
        "fixture_name",
        "home_team",
        "away_team",
        "home_team_normalized",
        "away_team_normalized",
        "start_time",
        "league_id",
        "market_id",
        "market",
        "line",
        "label",
    ]
    # Some columns may be missing in certain rows — fill fixture_name if absent
    for col in group_cols:
        if col not in df.columns:
            df[col] = ""

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(avg_odds=("odds", "mean"), best_odds=("odds", "max"), count=("odds", "size"))
        .reset_index()
    )

    # Split into Over and Under
    over = agg[agg["label"] == "Over"].rename(
        columns={"avg_odds": "over_avg", "best_odds": "over_best", "count": "over_count"}
    )
    under = agg[agg["label"] == "Under"].rename(
        columns={"avg_odds": "under_avg", "best_odds": "under_best", "count": "under_count"}
    )

    merge_keys = [
        "fixture_id",
        "fixture_name",
        "home_team",
        "away_team",
        "home_team_normalized",
        "away_team_normalized",
        "start_time",
        "league_id",
        "market_id",
        "market",
        "line",
    ]

    result = over[merge_keys + ["over_avg", "over_best", "over_count"]].merge(
        under[merge_keys + ["under_avg", "under_best", "under_count"]],
        on=merge_keys,
        how="outer",
    )

    # Ensure correct column order
    for col in KEEP_COLUMNS:
        if col not in result.columns:
            result[col] = None

    return result[KEEP_COLUMNS].sort_values(["league_id", "start_time", "line"]).reset_index(
        drop=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Save processed CSVs")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Corners ---
    corners_raw_path = RAW_DIR / "corners_raw.csv"
    if corners_raw_path.exists():
        logger.info(f"Loading {corners_raw_path}")
        raw_corners = pd.read_csv(corners_raw_path)
        logger.info(f"Raw corners: {len(raw_corners)} rows")
        corners = _process_corners(raw_corners, TARGET_LINES["corners"])
        logger.info(
            f"Processed corners: {len(corners)} rows, "
            f"{corners['fixture_id'].nunique()} fixtures, "
            f"lines: {sorted(corners['line'].unique())}"
        )
        if args.save:
            out_path = OUT_DIR / "corners_odds.csv"
            corners.to_csv(out_path, index=False)
            logger.info(f"Saved to {out_path}")
    else:
        logger.warning(f"Not found: {corners_raw_path}")

    # --- Cards ---
    cards_raw_path = RAW_DIR / "cards_raw.csv"
    if cards_raw_path.exists():
        logger.info(f"Loading {cards_raw_path}")
        raw_cards = pd.read_csv(cards_raw_path)
        logger.info(f"Raw cards: {len(raw_cards)} rows")
        cards = _process_cards(raw_cards, TARGET_LINES["cards"])
        logger.info(
            f"Processed cards: {len(cards)} rows, "
            f"{cards['fixture_id'].nunique()} fixtures, "
            f"lines: {sorted(cards['line'].unique())}"
        )

        # Compare with existing
        existing_path = OUT_DIR / "cards_odds.csv"
        if existing_path.exists():
            existing = pd.read_csv(existing_path)
            existing_target = existing[existing["line"].isin(TARGET_LINES["cards"])]
            logger.info(
                f"Existing cards at target lines: {len(existing_target)} rows, "
                f"{existing_target['fixture_id'].nunique()} fixtures"
            )

        if args.save:
            out_path = OUT_DIR / "cards_odds.csv"
            cards.to_csv(out_path, index=False)
            logger.info(f"Saved to {out_path}")
    else:
        logger.warning(f"Not found: {cards_raw_path}")


if __name__ == "__main__":
    main()
