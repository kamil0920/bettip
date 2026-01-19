#!/usr/bin/env python
"""
Fix xG Data Leakage

The current xG data is POST-MATCH xG from Understat, which causes data leakage.
This script creates proper PRE-MATCH xG features using rolling averages of historical xG.

Usage:
    python experiments/fix_xg_leakage.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime


def create_rolling_xg_features():
    """Create proper pre-match xG features using rolling averages."""
    print("=" * 70)
    print("FIXING xG DATA LEAKAGE")
    print("=" * 70)

    # Load features
    features_path = Path("data/03-features/features_all_5leagues_with_odds.csv")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} matches")

    # Check current xG columns
    xg_cols = [c for c in df.columns if c.startswith('xg_')]
    print(f"Current xG columns: {xg_cols}")

    # Load raw xG data
    xg_path = Path("data/03-features/xg_data_understat.csv")
    xg_df = pd.read_csv(xg_path)
    print(f"Loaded {len(xg_df)} xG records")

    # Parse dates
    xg_df['date'] = pd.to_datetime(xg_df['date'])
    xg_df = xg_df.sort_values('date').reset_index(drop=True)

    # Create team-level rolling xG averages
    print("\nCalculating rolling xG averages (window=10 matches)...")

    # For each team, calculate rolling average of xG for and against
    team_xg_history = {}

    # Process each match chronologically
    for _, row in xg_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        date = row['date']
        home_xg = row['home_xg']
        away_xg = row['away_xg']

        # Initialize team history if needed
        if home not in team_xg_history:
            team_xg_history[home] = {'xg_for': [], 'xg_against': []}
        if away not in team_xg_history:
            team_xg_history[away] = {'xg_for': [], 'xg_against': []}

        # Store match-level pre-match xG (rolling average BEFORE this match)
        # This is the proper pre-match estimate

        # Home team: average xG created and xG conceded in last N matches
        home_xg_for_avg = np.mean(team_xg_history[home]['xg_for'][-10:]) if team_xg_history[home]['xg_for'] else 1.3
        home_xg_against_avg = np.mean(team_xg_history[home]['xg_against'][-10:]) if team_xg_history[home]['xg_against'] else 1.3

        # Away team
        away_xg_for_avg = np.mean(team_xg_history[away]['xg_for'][-10:]) if team_xg_history[away]['xg_for'] else 1.3
        away_xg_against_avg = np.mean(team_xg_history[away]['xg_against'][-10:]) if team_xg_history[away]['xg_against'] else 1.3

        # Update history AFTER using it for prediction
        team_xg_history[home]['xg_for'].append(home_xg)
        team_xg_history[home]['xg_against'].append(away_xg)
        team_xg_history[away]['xg_for'].append(away_xg)
        team_xg_history[away]['xg_against'].append(home_xg)

    # Now create proper rolling features for the features file
    print("\nCreating pre-match xG features...")

    # Reset and recalculate
    team_xg_history = {}
    xg_features = []

    for _, row in xg_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_xg = row['home_xg']
        away_xg = row['away_xg']

        if home not in team_xg_history:
            team_xg_history[home] = {'xg_for': [], 'xg_against': []}
        if away not in team_xg_history:
            team_xg_history[away] = {'xg_for': [], 'xg_against': []}

        # PRE-MATCH estimates (before this match's data)
        home_xg_for_avg = np.mean(team_xg_history[home]['xg_for'][-10:]) if len(team_xg_history[home]['xg_for']) >= 3 else None
        home_xg_against_avg = np.mean(team_xg_history[home]['xg_against'][-10:]) if len(team_xg_history[home]['xg_against']) >= 3 else None
        away_xg_for_avg = np.mean(team_xg_history[away]['xg_for'][-10:]) if len(team_xg_history[away]['xg_for']) >= 3 else None
        away_xg_against_avg = np.mean(team_xg_history[away]['xg_against'][-10:]) if len(team_xg_history[away]['xg_against']) >= 3 else None

        # Expected xG for this match (pre-match estimate)
        # Home team xG = their attack strength vs away defense weakness
        # Away team xG = their attack strength vs home defense weakness
        if all([home_xg_for_avg, away_xg_against_avg]):
            expected_home_xg = (home_xg_for_avg + away_xg_against_avg) / 2
        else:
            expected_home_xg = None

        if all([away_xg_for_avg, home_xg_against_avg]):
            expected_away_xg = (away_xg_for_avg + home_xg_against_avg) / 2
        else:
            expected_away_xg = None

        xg_features.append({
            'home_team': home,
            'away_team': away,
            'date': row['date'],
            'match_id': row.get('match_id'),
            # Rolling averages (pre-match)
            'xg_home_attack_avg': home_xg_for_avg,
            'xg_home_defense_avg': home_xg_against_avg,
            'xg_away_attack_avg': away_xg_for_avg,
            'xg_away_defense_avg': away_xg_against_avg,
            # Expected xG for this match (pre-match estimate)
            'xg_home_expected': expected_home_xg,
            'xg_away_expected': expected_away_xg,
        })

        # Update history AFTER calculating features
        team_xg_history[home]['xg_for'].append(home_xg)
        team_xg_history[home]['xg_against'].append(away_xg)
        team_xg_history[away]['xg_for'].append(away_xg)
        team_xg_history[away]['xg_against'].append(home_xg)

    xg_features_df = pd.DataFrame(xg_features)

    # Add derived features
    xg_features_df['xg_total_expected'] = xg_features_df['xg_home_expected'] + xg_features_df['xg_away_expected']
    xg_features_df['xg_diff_expected'] = xg_features_df['xg_home_expected'] - xg_features_df['xg_away_expected']

    # BTTS probability from expected xG
    xg_features_df['xg_btts_expected'] = (
        (1 - np.exp(-xg_features_df['xg_home_expected'])) *
        (1 - np.exp(-xg_features_df['xg_away_expected']))
    )

    # Over 2.5 probability
    xg_features_df['xg_over25_expected'] = 1 - np.exp(-xg_features_df['xg_total_expected']) * (
        1 + xg_features_df['xg_total_expected'] + xg_features_df['xg_total_expected']**2/2
    )

    print(f"Created {len(xg_features_df)} pre-match xG feature records")
    print(f"Coverage: {xg_features_df['xg_home_expected'].notna().sum()}/{len(xg_features_df)}")

    # Now merge with features file using team normalizer
    from src.utils.team_normalizer import TeamNormalizer
    normalizer = TeamNormalizer()

    # Normalize team names in features
    df['home_team_normalized'] = df['home_team_name'].apply(
        lambda x: normalizer.to_understat(x) if pd.notna(x) else x
    )
    df['away_team_normalized'] = df['away_team_name'].apply(
        lambda x: normalizer.to_understat(x) if pd.notna(x) else x
    )

    # Parse dates
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date']).dt.date
    elif 'fixture_date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['fixture_date']).dt.date

    xg_features_df['date_parsed'] = xg_features_df['date'].dt.date

    # Create merge key
    df['merge_key'] = (
        df['home_team_normalized'] + '_' +
        df['away_team_normalized'] + '_' +
        df['date_parsed'].astype(str)
    )
    xg_features_df['merge_key'] = (
        xg_features_df['home_team'] + '_' +
        xg_features_df['away_team'] + '_' +
        xg_features_df['date_parsed'].astype(str)
    )

    # Remove old leaky xG columns
    old_xg_cols = [c for c in df.columns if c.startswith('xg_')]
    df = df.drop(columns=old_xg_cols)
    print(f"\nRemoved {len(old_xg_cols)} leaky xG columns")

    # Merge new proper xG features
    xg_merge_cols = [
        'merge_key',
        'xg_home_attack_avg', 'xg_home_defense_avg',
        'xg_away_attack_avg', 'xg_away_defense_avg',
        'xg_home_expected', 'xg_away_expected',
        'xg_total_expected', 'xg_diff_expected',
        'xg_btts_expected', 'xg_over25_expected'
    ]

    df = df.merge(
        xg_features_df[xg_merge_cols].drop_duplicates(subset=['merge_key']),
        on='merge_key',
        how='left'
    )

    # Clean up
    df = df.drop(columns=['merge_key', 'home_team_normalized', 'away_team_normalized', 'date_parsed'], errors='ignore')

    # Fill missing with defaults
    new_xg_cols = [c for c in df.columns if c.startswith('xg_')]
    for col in new_xg_cols:
        if 'attack' in col or 'expected' in col:
            df[col] = df[col].fillna(1.3)
        elif 'defense' in col:
            df[col] = df[col].fillna(1.3)
        elif 'btts' in col or 'over25' in col:
            df[col] = df[col].fillna(0.5)
        else:
            df[col] = df[col].fillna(0)

    # Coverage check
    coverage = (df['xg_home_expected'] != 1.3).sum()
    print(f"\nPre-match xG coverage: {coverage}/{len(df)} ({coverage/len(df)*100:.1f}%)")

    # Save
    df.to_csv(features_path, index=False)
    print(f"\nUpdated features file: {features_path}")

    # Verify no leakage
    print("\n" + "=" * 70)
    print("VERIFICATION: Checking for data leakage")
    print("=" * 70)

    # Correlation should be much lower now
    if 'home_goals' in df.columns:
        corr = df[['xg_home_expected', 'home_goals']].dropna().corr().iloc[0, 1]
        print(f"xg_home_expected vs home_goals correlation: {corr:.4f}")
        if corr > 0.5:
            print("WARNING: Correlation still high - may indicate issues")
        else:
            print("✓ Correlation reasonable for pre-match estimates")

    print("\n" + "=" * 70)
    print("DONE: xG features now use proper pre-match rolling averages")
    print("=" * 70)


if __name__ == "__main__":
    create_rolling_xg_features()
