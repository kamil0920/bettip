#!/usr/bin/env python3
"""Inject DynamicsFeatureEngineer outputs + league_cluster into existing features parquet.

Avoids full regenerate_all_features.py (~30 min) by bolting new columns onto existing file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.engineers.dynamics import DynamicsFeatureEngineer

FEATURES_PATH = Path("data/03-features/features_all_5leagues_with_odds.parquet")


def inject_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and merge dynamics features into existing dataframe."""
    eng = DynamicsFeatureEngineer(
        window=10, short_ema=5, long_ema=15,
        long_window=20, damping_factor=0.9, min_matches=3,
    )

    # Load match stats and build features directly (bypass create_features which needs 'matches' key)
    match_stats = eng._load_match_stats()
    if match_stats.empty:
        raise RuntimeError("No match_stats loaded — check data/01-raw/")

    match_stats = eng._derive_cards(match_stats)
    featured = eng._build_features(match_stats)

    # Extract only dynamics feature columns + fixture_id
    dyn_cols = [
        c for c in featured.columns
        if any(s in c for s in [
            '_skewness', '_kurtosis', '_cov',
            '_momentum_ratio', '_first_diff', '_variance_ratio',
        ])
    ]
    print(f"  Dynamics features generated: {len(dyn_cols)}")

    dynamics_df = featured[['fixture_id'] + dyn_cols].copy()

    # Drop any existing dynamics columns to avoid _x/_y
    existing_dyn = [c for c in df.columns if c in dyn_cols]
    if existing_dyn:
        print(f"  Dropping {len(existing_dyn)} existing dynamics cols to prevent collision")
        df = df.drop(columns=existing_dyn)

    merged = df.merge(dynamics_df, on='fixture_id', how='left')
    return merged


def inject_league_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Add league_cluster column via K-means (K=3) on league stat profiles."""
    if 'league_cluster' in df.columns:
        df = df.drop(columns=['league_cluster'])

    cluster_stat_cols = ['total_goals', 'total_fouls', 'total_cards', 'total_shots', 'total_corners']

    # Use home_win target directly if available (features parquet has it as target column)
    if 'home_win' in df.columns:
        df['_home_win'] = df['home_win'].astype(float)
    elif 'home_goals' in df.columns:
        df['_home_win'] = (df['home_goals'] > df['away_goals']).astype(float)

    available = [c for c in cluster_stat_cols if c in df.columns]
    if len(available) < 3 or '_home_win' not in df.columns:
        print("  WARNING: insufficient stats for clustering, defaulting to 0")
        df['league_cluster'] = 0
        return df

    # Per-league expanding averages (leakage-safe)
    league_avgs = {}
    for stat in available + ['_home_win']:
        league_avgs[stat] = df.groupby('league')[stat].transform(
            lambda x: x.expanding().mean().shift(1)
        )

    cluster_features = pd.DataFrame({s: league_avgs[s] for s in available + ['_home_win']})
    cluster_features['league'] = df['league'].values
    valid_mask = cluster_features[available + ['_home_win']].notna().all(axis=1)
    league_profiles = cluster_features[valid_mask].groupby('league').last().drop(columns=['league'], errors='ignore')

    if len(league_profiles) >= 3:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(league_profiles)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        cluster_map = dict(zip(league_profiles.index, kmeans.labels_))
        df['league_cluster'] = df['league'].map(cluster_map)
        mode_cluster = df['league_cluster'].mode().iloc[0] if not df['league_cluster'].mode().empty else 0
        df['league_cluster'] = df['league_cluster'].fillna(mode_cluster).astype(int)
        print(f"  League clusters: {cluster_map}")
    else:
        df['league_cluster'] = 0
        print("  WARNING: <3 leagues, defaulting to 0")

    # Cleanup
    df.drop(columns=['_home_win'], inplace=True, errors='ignore')
    print(f"  league_cluster values: {sorted(df['league_cluster'].unique())}")
    return df


def validate(df: pd.DataFrame, original_rows: int) -> None:
    """Validate the enhanced dataframe."""
    dyn_cols = [
        c for c in df.columns
        if any(s in c for s in [
            '_skewness', '_kurtosis', '_cov',
            '_momentum_ratio', '_first_diff', '_variance_ratio',
        ])
    ]

    # Check row count
    assert len(df) == original_rows, f"Row mismatch: {len(df)} vs {original_rows}"

    # Check dynamics features count
    print(f"\n--- Validation ---")
    print(f"  Rows: {len(df)}, Cols: {len(df.columns)}")
    print(f"  Dynamics features: {len(dyn_cols)}")
    print(f"  league_cluster: {'league_cluster' in df.columns}")

    # Check no _x/_y
    xy_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    assert len(xy_cols) == 0, f"_x/_y collision: {xy_cols}"
    print(f"  _x/_y columns: 0")

    # Check no inf
    inf_count = df[dyn_cols].apply(lambda x: x.isin([float('inf'), float('-inf')]).sum()).sum()
    assert inf_count == 0, f"Inf values found: {inf_count}"
    print(f"  Inf values: 0")

    # Coverage
    coverage = df[dyn_cols].notna().mean().mean() * 100
    print(f"  Dynamics coverage: {coverage:.1f}%")

    if len(dyn_cols) < 50:
        print(f"  WARNING: Only {len(dyn_cols)} dynamics features (expected ~56)")


def main():
    print(f"Loading {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    original_rows = len(df)
    print(f"  Shape: {df.shape}")

    print("\nStep 1: Injecting dynamics features...")
    df = inject_dynamics(df)

    print("\nStep 2: Adding league_cluster...")
    df = inject_league_cluster(df)

    validate(df, original_rows)

    print(f"\nSaving to {FEATURES_PATH}...")
    df.to_parquet(FEATURES_PATH, index=False)
    print("Done.")


if __name__ == '__main__':
    main()
