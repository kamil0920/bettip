#!/usr/bin/env python3
"""Post-merge odds injection: niche base odds + per-line odds + BTTS/HT estimates.

Called by collect-match-data.yaml after feature merge to ensure the parquet
contains all odds columns that models depend on.

Pipeline order:
  1. inject_niche_odds  — 16 base odds columns (theodds_*, HC, HT flat)
  2. sportmonks overlay — real per-line odds where available
  3. per_line_odds      — NB CDF ratio-scaled per-line odds (44 columns)
  4. BTTS per-line      — btts_yes_avg, btts_no_avg from expanding mean
  5. HT per-line        — totals_h1_{over,under}_avg_{0_5,1_5} from Poisson

Usage:
    python scripts/inject_all_odds.py                # Inject + save
    python scripts/inject_all_odds.py --upload        # Also upload to HF Hub
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PARQUET_PATH = Path("data/03-features/features_all_5leagues_with_odds.parquet")
MARGIN = 0.05


def inject_btts_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Add btts_yes_avg, btts_no_avg from per-league expanding mean + team factor."""
    if "btts" not in df.columns or "league" not in df.columns:
        logger.warning("Missing btts/league columns — skipping BTTS odds")
        return df

    df = df.sort_values("date").copy() if "date" in df.columns else df.copy()

    btts_rate = df.groupby("league")["btts"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    global_btts = df["btts"].expanding().mean().shift(1)
    btts_rate = btts_rate.fillna(global_btts).fillna(0.5).clip(0.15, 0.85)

    if "home_goals_scored_ema" in df.columns:
        attack = (
            df["home_goals_scored_ema"].fillna(1.3)
            + df["away_goals_scored_ema"].fillna(1.1)
        ) / 2.4
        attack = attack.clip(0.8, 1.2)
    else:
        attack = pd.Series(1.0, index=df.index)

    btts_yes_fair = (btts_rate * attack).clip(0.15, 0.85)
    btts_no_fair = 1 - btts_yes_fair

    for col, fair in [("btts_yes_avg", btts_yes_fair), ("btts_no_avg", btts_no_fair)]:
        odds = 1.0 / (fair * (1 + MARGIN))
        if col not in df.columns:
            df[col] = odds
        else:
            mask = df[col].isna()
            df.loc[mask, col] = odds[mask]

    logger.info(
        f"BTTS odds: yes median={df['btts_yes_avg'].median():.3f}, "
        f"no median={df['btts_no_avg'].median():.3f}"
    )
    return df


def inject_ht_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Add HT per-line odds from Poisson estimation."""
    from scipy.stats import poisson

    if "league" not in df.columns:
        logger.warning("Missing league column — skipping HT odds")
        return df

    df = df.sort_values("date").copy() if "date" in df.columns else df.copy()

    # Estimate HT total goals
    if "ht_home_goals" in df.columns and "ht_away_goals" in df.columns:
        ht_total = df["ht_home_goals"] + df["ht_away_goals"]
    elif "home_goals" in df.columns:
        ht_total = (df["home_goals"] + df["away_goals"]) * 0.42
    else:
        logger.warning("No goals columns — skipping HT odds")
        return df

    # Per-league expanding mean lambda with shift(1)
    ht_lambda = df.groupby("league").apply(
        lambda g: ht_total.loc[g.index].expanding().mean().shift(1),
        include_groups=False,
    )
    if hasattr(ht_lambda, "droplevel"):
        ht_lambda = ht_lambda.droplevel(0).sort_index()
    ht_lambda = ht_lambda.fillna(ht_total.expanding().mean().shift(1)).fillna(1.1)
    ht_lambda = ht_lambda.clip(0.5, 2.5)

    for line, suffix in [(0.5, "0_5"), (1.5, "1_5")]:
        p_over = (1 - poisson.cdf(int(line), ht_lambda)).clip(0.05, 0.95)
        p_under = poisson.cdf(int(line), ht_lambda).clip(0.05, 0.95)

        over_col = f"totals_h1_over_avg_{suffix}"
        under_col = f"totals_h1_under_avg_{suffix}"

        if over_col not in df.columns:
            df[over_col] = 1.0 / (p_over * (1 + MARGIN))
        if under_col not in df.columns:
            df[under_col] = 1.0 / (p_under * (1 + MARGIN))

        logger.info(
            f"HT {suffix}: over median={df[over_col].median():.3f}, "
            f"under median={df[under_col].median():.3f}"
        )

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="Upload to HF Hub")
    args = parser.parse_args()

    if not PARQUET_PATH.exists():
        logger.error(f"Parquet not found: {PARQUET_PATH}")
        return

    df = pd.read_parquet(PARQUET_PATH)
    logger.info(f"Loaded {PARQUET_PATH}: {df.shape}")

    # Step 1: Niche base odds (theodds_*, HC, HT flat)
    sys.path.insert(0, str(Path(__file__).parent))
    from inject_niche_odds import inject_niche_odds

    df = inject_niche_odds(df)
    logger.info(f"After niche odds: {df.shape}")

    # Step 2: Sportmonks overlay (real per-line odds where available)
    try:
        from src.odds.sportmonks_per_line import overlay_sportmonks_per_line_odds

        df = overlay_sportmonks_per_line_odds(df)
        logger.info(f"After Sportmonks overlay: {df.shape}")
    except Exception as e:
        logger.warning(f"Sportmonks overlay skipped: {e}")

    # Step 3: NB CDF per-line odds
    from src.odds.per_line_odds import generate_per_line_odds

    n_before = len(df.columns)
    df = generate_per_line_odds(df)
    logger.info(f"After per-line odds: {df.shape} (+{len(df.columns) - n_before} cols)")

    # Step 4: BTTS per-line odds
    df = inject_btts_odds(df)

    # Step 5: HT per-line odds
    df = inject_ht_odds(df)

    # Save
    df.to_parquet(PARQUET_PATH, index=False)
    logger.info(f"Saved: {PARQUET_PATH} ({df.shape})")

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
