#!/usr/bin/env python
"""
xG Data Integration Script

Integrates Understat xG data into the main features file using team name normalization.

Usage:
    python experiments/integrate_xg_data.py status    # Check xG data coverage
    python experiments/integrate_xg_data.py merge     # Merge xG into features
    python experiments/integrate_xg_data.py validate  # Validate merge quality
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.team_normalizer import TeamNormalizer

# Paths
XG_DATA_PATH = Path("data/03-features/xg_data_understat.csv")
FEATURES_PATH = Path("data/03-features/features_all_5leagues_with_odds.csv")


def check_xg_status():
    """Check xG data coverage and matching rates."""
    print("=" * 60)
    print("xG Data Status")
    print("=" * 60)

    # Load xG data
    if not XG_DATA_PATH.exists():
        print(f"xG data not found at {XG_DATA_PATH}")
        return

    xg_df = pd.read_csv(XG_DATA_PATH)
    print(f"xG records: {len(xg_df)}")
    print(f"Leagues: {xg_df['league'].unique().tolist()}")
    print(f"Seasons: {sorted(xg_df['season'].unique().tolist())}")

    # Load features
    if not FEATURES_PATH.exists():
        print(f"\nFeatures file not found at {FEATURES_PATH}")
        return

    features_df = pd.read_csv(FEATURES_PATH)
    print(f"\nFeatures file: {len(features_df)} rows")

    # Check existing xG columns
    xg_cols = [c for c in features_df.columns if 'xg' in c.lower()]
    print(f"Existing xG columns: {len(xg_cols)}")
    if xg_cols:
        print(f"  Columns: {xg_cols[:10]}")

    # Initialize normalizer
    normalizer = TeamNormalizer()

    # Normalize team names in features
    features_df['home_team_normalized'] = features_df['home_team_name'].apply(
        lambda x: normalizer.to_understat(x) if pd.notna(x) else x
    )
    features_df['away_team_normalized'] = features_df['away_team_name'].apply(
        lambda x: normalizer.to_understat(x) if pd.notna(x) else x
    )

    # Convert dates for matching
    xg_df['date'] = pd.to_datetime(xg_df['date']).dt.date
    if 'date' in features_df.columns:
        features_df['date_parsed'] = pd.to_datetime(features_df['date']).dt.date
    elif 'fixture_date' in features_df.columns:
        features_df['date_parsed'] = pd.to_datetime(features_df['fixture_date']).dt.date
    else:
        print("Warning: No date column found in features")
        return

    # Try to match on date + teams
    matched_count = 0
    unmatched_teams = set()

    for _, row in features_df.iterrows():
        home = row['home_team_normalized']
        away = row['away_team_normalized']
        date = row['date_parsed']

        match = xg_df[
            (xg_df['home_team'] == home) &
            (xg_df['away_team'] == away) &
            (xg_df['date'] == date)
        ]

        if len(match) > 0:
            matched_count += 1
        else:
            unmatched_teams.add(home)
            unmatched_teams.add(away)

    print(f"\n--- Matching Results ---")
    print(f"Matched: {matched_count}/{len(features_df)} ({matched_count/len(features_df)*100:.1f}%)")
    print(f"Unmatched teams ({len(unmatched_teams)}): {sorted(list(unmatched_teams))[:20]}...")


def merge_xg_data():
    """Merge xG data into features file."""
    print("=" * 60)
    print("Merging xG Data into Features")
    print("=" * 60)

    # Load data
    if not XG_DATA_PATH.exists():
        print(f"xG data not found at {XG_DATA_PATH}")
        return

    xg_df = pd.read_csv(XG_DATA_PATH)
    print(f"Loaded {len(xg_df)} xG records")

    if not FEATURES_PATH.exists():
        print(f"Features file not found at {FEATURES_PATH}")
        return

    features_df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(features_df)} feature rows")

    # Check if xG already integrated
    existing_xg_cols = [c for c in features_df.columns if c.startswith('xg_')]
    if existing_xg_cols:
        print(f"Warning: xG columns already exist: {existing_xg_cols}")
        features_df = features_df.drop(columns=existing_xg_cols)

    # Initialize normalizer
    normalizer = TeamNormalizer()

    # Normalize team names in features
    features_df['home_team_normalized'] = features_df['home_team_name'].apply(
        lambda x: normalizer.to_understat(x) if pd.notna(x) else x
    )
    features_df['away_team_normalized'] = features_df['away_team_name'].apply(
        lambda x: normalizer.to_understat(x) if pd.notna(x) else x
    )

    # Convert dates
    xg_df['date'] = pd.to_datetime(xg_df['date']).dt.date
    if 'date' in features_df.columns:
        features_df['date_parsed'] = pd.to_datetime(features_df['date']).dt.date
    elif 'fixture_date' in features_df.columns:
        features_df['date_parsed'] = pd.to_datetime(features_df['fixture_date']).dt.date
    else:
        print("Error: No date column found")
        return

    # Create merge key
    xg_df['merge_key'] = (
        xg_df['home_team'] + '_' +
        xg_df['away_team'] + '_' +
        xg_df['date'].astype(str)
    )
    features_df['merge_key'] = (
        features_df['home_team_normalized'] + '_' +
        features_df['away_team_normalized'] + '_' +
        features_df['date_parsed'].astype(str)
    )

    # Prepare xG features
    xg_features = xg_df[['merge_key', 'home_xg', 'away_xg']].copy()
    xg_features = xg_features.rename(columns={
        'home_xg': 'xg_home_pre',
        'away_xg': 'xg_away_pre'
    })
    xg_features = xg_features.drop_duplicates(subset=['merge_key'])

    # Merge
    merged = features_df.merge(xg_features, on='merge_key', how='left')

    # Create derived xG features
    merged['xg_total_pre'] = merged['xg_home_pre'] + merged['xg_away_pre']
    merged['xg_diff_pre'] = merged['xg_home_pre'] - merged['xg_away_pre']
    merged['xg_home_ratio'] = merged['xg_home_pre'] / merged['xg_total_pre'].replace(0, np.nan)

    # xG-based probabilities (simple Poisson approximation)
    merged['xg_btts_prob'] = (
        (1 - np.exp(-merged['xg_home_pre'])) *
        (1 - np.exp(-merged['xg_away_pre']))
    )
    merged['xg_over25_prob'] = 1 - np.exp(-merged['xg_total_pre']) * (
        1 + merged['xg_total_pre'] + merged['xg_total_pre']**2/2
    )
    merged['xg_under25_prob'] = 1 - merged['xg_over25_prob']

    # Clean up
    merged = merged.drop(columns=['merge_key', 'home_team_normalized', 'away_team_normalized', 'date_parsed'])

    # Fill missing xG with league averages
    xg_cols = [c for c in merged.columns if c.startswith('xg_')]
    for col in xg_cols:
        if col in ['xg_home_pre', 'xg_away_pre']:
            merged[col] = merged[col].fillna(1.3)  # League average
        elif col == 'xg_total_pre':
            merged[col] = merged[col].fillna(2.6)
        elif col == 'xg_diff_pre':
            merged[col] = merged[col].fillna(0)
        elif col == 'xg_home_ratio':
            merged[col] = merged[col].fillna(0.5)
        elif col == 'xg_btts_prob':
            merged[col] = merged[col].fillna(0.5)
        elif col in ['xg_over25_prob', 'xg_under25_prob']:
            merged[col] = merged[col].fillna(0.5)

    # Calculate coverage
    coverage = (merged['xg_home_pre'] != 1.3).sum()  # Non-default values
    print(f"\nxG coverage: {coverage}/{len(merged)} ({coverage/len(merged)*100:.1f}%)")

    # Save
    output_path = FEATURES_PATH.with_name('features_all_5leagues_with_xg.csv')
    merged.to_csv(output_path, index=False)
    print(f"Saved features with xG to: {output_path}")

    # Also update main features file
    merged.to_csv(FEATURES_PATH, index=False)
    print(f"Updated main features file: {FEATURES_PATH}")

    # Summary
    print(f"\n--- xG Features Added ---")
    xg_cols = [c for c in merged.columns if c.startswith('xg_')]
    for col in xg_cols:
        print(f"  {col}: mean={merged[col].mean():.3f}, std={merged[col].std():.3f}")


def validate_merge():
    """Validate xG merge quality."""
    print("=" * 60)
    print("Validating xG Merge")
    print("=" * 60)

    if not FEATURES_PATH.exists():
        print(f"Features file not found at {FEATURES_PATH}")
        return

    df = pd.read_csv(FEATURES_PATH)

    # Check xG columns
    xg_cols = [c for c in df.columns if c.startswith('xg_')]
    if not xg_cols:
        print("No xG columns found! Run 'merge' first.")
        return

    print(f"xG columns found: {len(xg_cols)}")
    for col in xg_cols:
        non_default = (df[col].notna() & (df[col] != 1.3) & (df[col] != 0.5)).sum()
        print(f"  {col}: {non_default}/{len(df)} non-default ({non_default/len(df)*100:.1f}%)")

    # Correlation with actual goals
    if 'home_goals' in df.columns and 'xg_home_pre' in df.columns:
        corr_home = df[['xg_home_pre', 'home_goals']].dropna().corr().iloc[0, 1]
        corr_away = df[['xg_away_pre', 'away_goals']].dropna().corr().iloc[0, 1]
        print(f"\nCorrelation with actual goals:")
        print(f"  Home xG vs home goals: {corr_home:.3f}")
        print(f"  Away xG vs away goals: {corr_away:.3f}")

    # Check for data leakage (xG should be from BEFORE match)
    # xG data from Understat is post-match, so we need to check if we're using
    # pre-match data correctly
    print("\nNote: Understat xG is POST-MATCH data (actual xG from the match).")
    print("For true pre-match predictions, use rolling averages of past xG.")


def main():
    parser = argparse.ArgumentParser(description='xG Data Integration')
    parser.add_argument('action', choices=['status', 'merge', 'validate'],
                       help='Action to perform')
    args = parser.parse_args()

    if args.action == 'status':
        check_xg_status()
    elif args.action == 'merge':
        merge_xg_data()
    elif args.action == 'validate':
        validate_merge()


if __name__ == "__main__":
    main()
