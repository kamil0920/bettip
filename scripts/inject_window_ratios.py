#!/usr/bin/env python3
"""Inject WindowRatioFeatureEngineer outputs into existing features parquet.

Avoids full regenerate_all_features.py (~30 min) by bolting new columns onto existing file.
Same pattern as inject_dynamics_features.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.engineers.window_ratio import WindowRatioFeatureEngineer

FEATURES_PATH = Path("data/03-features/features_all_5leagues_with_odds.parquet")


def inject_window_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and merge window ratio features into existing dataframe."""
    eng = WindowRatioFeatureEngineer(short_ema=3, long_ema=12, min_matches=3)

    # Load match stats and build features directly
    match_stats = eng._load_match_stats()
    if match_stats.empty:
        raise RuntimeError("No match_stats loaded — check data/01-raw/")

    featured = eng._build_features(match_stats)

    # Extract only ratio feature columns + fixture_id
    ratio_cols = [
        c for c in featured.columns
        if c.endswith('_ratio') or c.endswith('_ratio_diff')
        if '_momentum_ratio' not in c and '_variance_ratio' not in c
    ]
    print(f"  Window ratio features generated: {len(ratio_cols)}")

    ratio_df = featured[['fixture_id'] + ratio_cols].copy()

    # Drop any existing ratio columns to avoid _x/_y
    existing = [c for c in df.columns if c in ratio_cols]
    if existing:
        print(f"  Dropping {len(existing)} existing ratio cols to prevent collision")
        df = df.drop(columns=existing)

    merged = df.merge(ratio_df, on='fixture_id', how='left')
    return merged


def validate(df: pd.DataFrame, original_rows: int) -> None:
    """Validate the enhanced dataframe."""
    _OUR_STATS = ['goals', 'goals_conceded', 'points', 'shots_on_target', 'possession', 'goals_per_shot']
    ratio_cols = [
        c for c in df.columns
        if any(
            c == f'{side}_{stat}_ratio' or c == f'{stat}_ratio_diff'
            for side in ('home', 'away')
            for stat in _OUR_STATS
        )
    ]

    assert len(df) == original_rows, f"Row mismatch: {len(df)} vs {original_rows}"

    print(f"\n--- Validation ---")
    print(f"  Rows: {len(df)}, Cols: {len(df.columns)}")
    print(f"  Window ratio features: {len(ratio_cols)}")
    for c in sorted(ratio_cols):
        print(f"    {c}")

    # No _x/_y
    xy_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    assert len(xy_cols) == 0, f"_x/_y collision: {xy_cols}"
    print(f"  _x/_y columns: 0")

    # No inf
    inf_count = df[ratio_cols].apply(lambda x: x.isin([float('inf'), float('-inf')]).sum()).sum()
    assert inf_count == 0, f"Inf values found: {inf_count}"
    print(f"  Inf values: 0")

    # Coverage
    coverage = df[ratio_cols].notna().mean()
    print(f"  Coverage per feature:")
    for c in sorted(ratio_cols):
        print(f"    {c}: {coverage[c]*100:.1f}%")
    print(f"  Avg coverage: {coverage.mean()*100:.1f}%")


def main():
    print(f"Loading {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    original_rows = len(df)
    print(f"  Shape: {df.shape}")

    print("\nInjecting window ratio features...")
    df = inject_window_ratios(df)

    validate(df, original_rows)

    print(f"\nSaving to {FEATURES_PATH}...")
    df.to_parquet(FEATURES_PATH, index=False)
    print("Done.")


if __name__ == '__main__':
    main()
