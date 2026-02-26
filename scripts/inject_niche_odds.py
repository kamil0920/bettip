#!/usr/bin/env python3
"""Inject realistic niche odds columns into the features parquet.

All niche markets (cards, corners, shots, fouls, HC, HT) currently have ZERO odds
columns in the features parquet, causing the sniper optimizer to fall back to
np.full(len(df), 2.5). Real niche odds are ~1.80-2.05, meaning thresholds are
calibrated for 67% higher profit per win than reality.

This script adds 16 estimated odds columns using the same market-efficiency +
team-factor pattern as the existing odds loaders.

Usage:
    python scripts/inject_niche_odds.py              # Dry-run: show stats
    python scripts/inject_niche_odds.py --save       # Save to parquet
    python scripts/inject_niche_odds.py --upload     # Upload parquet to HF Hub
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PARQUET_PATH = Path("data/03-features/features_all_5leagues_with_odds.parquet")

# Bookmaker margin: 5% overround (total implied prob = 1.05)
MARGIN = 0.05

# Default base odds from existing loaders — represent typical niche market pricing.
# Each odds column is shared across ALL line variants (e.g. cards_under_15..65),
# so we use "generic niche market" defaults, NOT line-specific fair prices.
BASE_ODDS = {
    "theodds_cards": (1.85, 1.95),  # from cards_odds_loader.py
    "theodds_corners": (1.90, 1.90),  # from corners_odds_loader.py
    "theodds_shots": (1.90, 1.90),  # from shots_odds_loader.py
    "fouls": (1.90, 1.90),  # same pattern as corners/shots
}


def _set_base_odds(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Set base over/under odds from default bookmaker pricing."""
    base_over, base_under = BASE_ODDS[prefix]
    df[f"{prefix}_over_odds"] = base_over
    df[f"{prefix}_under_odds"] = base_under
    logger.info(f"{prefix}: base over={base_over:.3f}, under={base_under:.3f}")
    return df


def _apply_team_factor(
    df: pd.DataFrame,
    prefix: str,
    home_col: str,
    away_col: str,
    divisor: float,
) -> pd.DataFrame:
    """Add per-match variation from team EMA stats (clip 0.8-1.2)."""
    if home_col in df.columns and away_col in df.columns:
        factor = (df[home_col] + df[away_col]) / divisor
        factor = factor.clip(0.8, 1.2)
        df[f"{prefix}_over_odds"] = df[f"{prefix}_over_odds"] / factor
        df[f"{prefix}_under_odds"] = df[f"{prefix}_under_odds"] * factor
    return df


def inject_niche_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Add 16 niche odds columns to the features DataFrame."""
    orig_cols = len(df.columns)

    # --- Cards ---
    # Divisor = median(home_avg_yellows + away_avg_yellows) ≈ 3.4
    df = _set_base_odds(df, "theodds_cards")
    df = _apply_team_factor(
        df, "theodds_cards", "home_avg_yellows", "away_avg_yellows", divisor=3.4
    )
    # Rename under column to match sniper config expectations
    df["cards_under_odds"] = df["theodds_cards_under_odds"]
    df.drop(columns=["theodds_cards_under_odds"], inplace=True)

    # --- Corners ---
    df = _set_base_odds(df, "theodds_corners")
    df = _apply_team_factor(
        df,
        "theodds_corners",
        "home_corners_won_ema",
        "away_corners_won_ema",
        divisor=10,
    )
    df["corners_under_odds"] = df["theodds_corners_under_odds"]
    df.drop(columns=["theodds_corners_under_odds"], inplace=True)

    # --- Shots ---
    # Divisor = median(home_shots_ema + away_shots_ema) ≈ 24.5
    df = _set_base_odds(df, "theodds_shots")
    df = _apply_team_factor(
        df, "theodds_shots", "home_shots_ema", "away_shots_ema", divisor=24.5
    )
    df["shots_under_odds"] = df["theodds_shots_under_odds"]
    df.drop(columns=["theodds_shots_under_odds"], inplace=True)

    # --- Fouls (no existing loader, same pattern) ---
    # Divisor = median(home_fouls_committed_ema + away_fouls_committed_ema) ≈ 23.6
    df = _set_base_odds(df, "fouls")
    df = _apply_team_factor(
        df,
        "fouls",
        "home_fouls_committed_ema",
        "away_fouls_committed_ema",
        divisor=23.6,
    )

    # --- HC: Corners handicap (flat estimated odds from API lookup medians) ---
    df["cornershc_over_odds"] = 1.99
    df["cornershc_under_odds"] = 1.85
    logger.info("cornershc: flat over=1.990, under=1.850")

    # --- HC: Cards handicap ---
    df["cardshc_over_odds"] = 1.96
    df["cardshc_under_odds"] = 1.79
    logger.info("cardshc: flat over=1.960, under=1.790")

    # --- HT: Half-time H2H odds (from base rates + 5% margin) ---
    hw_rate = 0.342  # home_win_h1 rate from data
    aw_rate = 0.262  # away_win_h1 rate from data
    df["h2h_h1_home_avg"] = 1 / (hw_rate + MARGIN * hw_rate)
    df["h2h_h1_away_avg"] = 1 / (aw_rate + MARGIN * aw_rate)
    logger.info(
        f"HT H2H: home={df['h2h_h1_home_avg'].iloc[0]:.3f}, "
        f"away={df['h2h_h1_away_avg'].iloc[0]:.3f}"
    )

    # --- HT: Totals odds (over/under 0.5 goals in first half) ---
    ht_over_05_rate = 0.720
    ht_under_05_rate = 1 - ht_over_05_rate
    df["totals_h1_over_odds"] = 1 / (
        ht_over_05_rate + MARGIN * ht_over_05_rate / (ht_over_05_rate + ht_under_05_rate)
    )
    df["totals_h1_under_odds"] = 1 / (
        ht_under_05_rate + MARGIN * ht_under_05_rate / (ht_over_05_rate + ht_under_05_rate)
    )
    logger.info(
        f"HT totals: over_05={df['totals_h1_over_odds'].iloc[0]:.3f}, "
        f"under_05={df['totals_h1_under_odds'].iloc[0]:.3f}"
    )

    new_cols = len(df.columns) - orig_cols
    logger.info(f"Added {new_cols} niche odds columns ({orig_cols} -> {len(df.columns)})")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Save updated parquet")
    parser.add_argument("--upload", action="store_true", help="Upload to HF Hub")
    args = parser.parse_args()

    logger.info(f"Loading {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    logger.info(f"Shape before: {df.shape}")

    # Check for existing niche odds columns (skip if already present)
    expected_cols = [
        "theodds_cards_over_odds", "cards_under_odds",
        "theodds_corners_over_odds", "corners_under_odds",
        "theodds_shots_over_odds", "shots_under_odds",
        "fouls_over_odds", "fouls_under_odds",
        "cornershc_over_odds", "cornershc_under_odds",
        "cardshc_over_odds", "cardshc_under_odds",
        "h2h_h1_home_avg", "h2h_h1_away_avg",
        "totals_h1_over_odds", "totals_h1_under_odds",
    ]
    existing = [c for c in expected_cols if c in df.columns]
    if existing:
        logger.warning(f"Dropping {len(existing)} existing niche odds columns: {existing}")
        df.drop(columns=existing, inplace=True)

    df = inject_niche_odds(df)

    # Verify all expected columns present
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after injection: {missing}")

    # Summary stats
    logger.info(f"Shape after: {df.shape}")
    logger.info("\nOdds summary:")
    for col in sorted(expected_cols):
        vals = df[col].dropna()
        logger.info(f"  {col}: median={vals.median():.3f}, mean={vals.mean():.3f}, "
                     f"std={vals.std():.3f}, coverage={len(vals)/len(df):.1%}")

    if args.save:
        df.to_parquet(PARQUET_PATH, index=False)
        logger.info(f"Saved to {PARQUET_PATH}")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        token = os.environ["HF_TOKEN"]
        repo_id = os.environ.get("HF_REPO_ID", "czlowiekZplanety/bettip-data")
        api.upload_file(
            path_or_fileobj=str(PARQUET_PATH),
            path_in_repo=str(PARQUET_PATH),
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        logger.info(f"Uploaded to HF Hub: {repo_id}")


if __name__ == "__main__":
    main()
