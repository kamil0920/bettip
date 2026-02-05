#!/usr/bin/env python3
"""
Regenerate all features for all leagues with niche market EMAs.

This script:
1. Runs feature engineering for each league
2. Merges betting odds
3. Combines all league features into features_all_5leagues_with_odds.csv
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

LEAGUES = [
    'premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1',
    'ekstraklasa',
    'eredivisie', 'portuguese_liga', 'turkish_super_lig',
    'belgian_pro_league', 'scottish_premiership',
]


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"WARNING: {description} failed with code {result.returncode}")
        return False
    return True


def main():
    features_dir = PROJECT_ROOT / 'data' / '03-features'
    features_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate features for each league
    print("\n" + "="*60)
    print("STEP 1: Generate features for each league")
    print("="*60)

    for league in LEAGUES:
        config_file = PROJECT_ROOT / 'config' / f'{league}.yaml'
        if not config_file.exists():
            print(f"Config not found for {league}, skipping...")
            continue

        run_command(
            ['uv', 'run', 'python', 'entrypoints/features.py',
             '--config', str(config_file),
             '--output', f'features_{league}.csv'],
            f"Generating features for {league}"
        )

    # Step 2: Fetch odds for each league
    print("\n" + "="*60)
    print("STEP 2: Fetch and merge betting odds")
    print("="*60)

    for league in LEAGUES:
        run_command(
            ['uv', 'run', 'python', 'entrypoints/fetch_odds.py',
             '--league', league,
             '--seasons', '2020', '2021', '2022', '2023', '2024',
             '--features-dir', 'data/03-features',
             '--cache-dir', 'data/odds-cache',
             '--output-suffix', '_with_odds'],
            f"Fetching odds for {league}"
        )

    # Step 3: Combine all features
    print("\n" + "="*60)
    print("STEP 3: Combining all league features")
    print("="*60)

    all_features = []
    import os

    for league in LEAGUES:
        with_odds = features_dir / f'features_{league}_with_odds.csv'
        without_odds = features_dir / f'features_{league}.csv'

        if with_odds.exists() and without_odds.exists():
            if os.path.getmtime(with_odds) >= os.path.getmtime(without_odds):
                csv_file = with_odds
            else:
                csv_file = without_odds
        elif with_odds.exists():
            csv_file = with_odds
        elif without_odds.exists():
            csv_file = without_odds
        else:
            print(f"No features found for {league}")
            continue

        print(f"Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        df['league'] = league
        all_features.append(df)
        print(f"  {len(df)} rows, {len(df.columns)} columns")

    if all_features:
        merged = pd.concat(all_features, ignore_index=True)

        # Calculate btts target if not present
        if 'btts' not in merged.columns and 'home_goals' in merged.columns:
            merged['btts'] = ((merged['home_goals'] > 0) & (merged['away_goals'] > 0)).astype(int)

        # Save combined file (Parquet primary + CSV for backward compat)
        from src.utils.data_io import save_features
        output_path = features_dir / 'features_all_5leagues_with_odds'
        save_features(merged, output_path, dual_format=True)
        print(f"\nSaved combined features to {output_path}")
        print(f"Total: {len(merged)} rows, {len(merged.columns)} columns")

        # Show niche market features
        niche_cols = [c for c in merged.columns if any(x in c.lower() for x in
                     ['corner', 'foul', 'shot', 'card'])]
        print(f"\nNiche market features found: {len(niche_cols)}")
        for col in niche_cols[:20]:
            non_null = merged[col].notna().sum()
            print(f"  {col}: {non_null} non-null ({non_null/len(merged)*100:.1f}%)")
    else:
        print("ERROR: No features generated!")
        sys.exit(1)


if __name__ == '__main__':
    main()
