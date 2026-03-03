#!/usr/bin/env python3
"""Overlay real Sportmonks per-line odds + NB CDF fill into features parquet.

Runs after inject_niche_odds.py (which adds default-line odds columns).
Uses the same overlay pattern as regenerate_all_features.py:210-231.

Usage:
    python scripts/overlay_per_line_odds.py              # Dry-run: show stats
    python scripts/overlay_per_line_odds.py --save       # Save to parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PARQUET_PATH = Path("data/03-features/features_all_5leagues_with_odds.parquet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Save updated parquet")
    args = parser.parse_args()

    logger.info(f"Loading {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    logger.info(f"Shape before: {df.shape}")

    # Step 1: Overlay real Sportmonks per-line odds
    from src.odds.sportmonks_per_line import overlay_sportmonks_per_line_odds

    df = overlay_sportmonks_per_line_odds(df)

    # Step 2: Fill remaining gaps with NB CDF estimates
    from src.odds.per_line_odds import generate_per_line_odds

    n_cols_before = len(df.columns)
    df = generate_per_line_odds(df)
    n_new = len(df.columns) - n_cols_before
    logger.info(f"Added {n_new} per-line odds columns via NB CDF fill")

    # Summary stats for per-line odds columns
    avg_cols = sorted(c for c in df.columns if "_avg_" in c)
    logger.info(f"\nPer-line odds columns ({len(avg_cols)}):")
    for col in avg_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            logger.info(
                f"  {col}: mean={vals.mean():.3f}, median={vals.median():.3f}, "
                f"coverage={len(vals)/len(df):.1%}"
            )

    logger.info(f"Shape after: {df.shape}")

    if args.save:
        df.to_parquet(PARQUET_PATH, index=False)
        logger.info(f"Saved to {PARQUET_PATH}")


if __name__ == "__main__":
    main()
