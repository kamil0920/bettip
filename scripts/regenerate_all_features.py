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

        # League features — one-hot encoding
        league_dummies = pd.get_dummies(merged['league'], prefix='league_is', dtype=int)
        merged = pd.concat([merged, league_dummies], axis=1)
        print(f"  Added {len(league_dummies.columns)} league one-hot columns")

        # League features — expanding averages for niche stats
        merged = merged.sort_values('date').reset_index(drop=True)
        niche_stats = ['total_fouls', 'total_cards', 'total_shots', 'total_corners']
        added_avg = 0
        for stat in niche_stats:
            if stat in merged.columns:
                col_name = f'league_avg_{stat}'
                merged[col_name] = (
                    merged.groupby('league')[stat]
                    .transform(lambda x: x.expanding().mean().shift(1))
                )
                global_mean = merged[stat].mean()
                merged[col_name] = merged[col_name].fillna(global_mean)
                added_avg += 1
                print(f"  {col_name}: range {merged[col_name].min():.1f} - {merged[col_name].max():.1f}")
        print(f"  Added {added_avg} league expanding average columns")

        # League clustering (K=3 from summary stats)
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        cluster_stat_cols = ['total_goals', 'total_fouls', 'total_cards', 'total_shots', 'total_corners']
        # Derive total_goals if not present
        if 'total_goals' not in merged.columns and 'home_goals' in merged.columns:
            merged['total_goals'] = merged['home_goals'] + merged['away_goals']
        # Derive home_win indicator for cluster stats
        if 'home_goals' in merged.columns and 'away_goals' in merged.columns:
            merged['_home_win'] = (merged['home_goals'] > merged['away_goals']).astype(float)

        available_cluster_stats = [c for c in cluster_stat_cols if c in merged.columns]
        if len(available_cluster_stats) >= 3 and '_home_win' in merged.columns:
            # Per-league expanding averages (leakage-safe via shift(1))
            league_avgs = {}
            for stat in available_cluster_stats + ['_home_win']:
                league_avgs[stat] = (
                    merged.groupby('league')[stat]
                    .transform(lambda x: x.expanding().mean().shift(1))
                )

            # Build per-league summary at each row (use the expanding averages)
            cluster_features = pd.DataFrame({
                stat: league_avgs[stat] for stat in available_cluster_stats + ['_home_win']
            })

            # Drop rows with NaN (first match per league)
            valid_mask = cluster_features.notna().all(axis=1)

            # Get unique league profiles (last valid row per league for cluster fitting)
            cluster_features['league'] = merged['league']
            league_profiles = cluster_features[valid_mask].groupby('league').last()

            if len(league_profiles) >= 3:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(league_profiles)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                league_cluster_map = dict(zip(league_profiles.index, kmeans.labels_))

                merged['league_cluster'] = merged['league'].map(league_cluster_map)
                # Fill NaN for leagues that didn't have enough data
                mode_cluster = merged['league_cluster'].mode().iloc[0] if not merged['league_cluster'].mode().empty else 0
                merged['league_cluster'] = merged['league_cluster'].fillna(mode_cluster).astype(int)
                print(f"  League clusters: {league_cluster_map}")
            else:
                merged['league_cluster'] = 0
                print("  WARNING: Not enough leagues for clustering, defaulting to 0")

            cluster_features.drop(columns=['league'], inplace=True)
        else:
            merged['league_cluster'] = 0
            print(f"  WARNING: Missing cluster stats ({available_cluster_stats}), defaulting to 0")

        # Clean up temp column
        if '_home_win' in merged.columns:
            merged.drop(columns=['_home_win'], inplace=True)

        print(f"  Added league_cluster column (values: {sorted(merged['league_cluster'].unique())})")

        # Calculate btts target if not present
        if 'btts' not in merged.columns and 'home_goals' in merged.columns:
            merged['btts'] = ((merged['home_goals'] > 0) & (merged['away_goals'] > 0)).astype(int)

        # Calculate over25/under25 targets from total_goals
        if 'home_goals' in merged.columns:
            total_goals = merged['home_goals'] + merged['away_goals']
            if 'over25' not in merged.columns:
                merged['over25'] = (total_goals > 2.5).astype(int)
                print(f"  Derived over25: {merged['over25'].sum()} positive ({merged['over25'].mean()*100:.1f}%)")
            if 'under25' not in merged.columns:
                merged['under25'] = (total_goals <= 2.5).astype(int)
                print(f"  Derived under25: {merged['under25'].sum()} positive ({merged['under25'].mean()*100:.1f}%)")

        # Save combined file (Parquet only — CSV removed to save space)
        output_path = features_dir / 'features_all_5leagues_with_odds.parquet'
        merged.to_parquet(output_path, index=False)
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
