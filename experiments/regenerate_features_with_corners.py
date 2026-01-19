#!/usr/bin/env python
"""
Regenerate Features with Corner/Shots EMAs

Adds corner and shots features from match_stats to the main features file.
This uses the collected match_stats for all leagues where available.

Usage:
    python experiments/regenerate_features_with_corners.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict


def load_all_match_stats() -> pd.DataFrame:
    """Load match_stats from all leagues/seasons."""
    leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']
    seasons = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']

    all_stats = []
    for league in leagues:
        for season in seasons:
            stats_path = Path(f"data/01-raw/{league}/{season}/match_stats.parquet")
            if stats_path.exists():
                df = pd.read_parquet(stats_path)
                df['league'] = league
                df['season'] = season
                all_stats.append(df)
                print(f"  {league}/{season}: {len(df)} matches")

    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        print(f"\nTotal match_stats: {len(combined)} matches")
        return combined
    return pd.DataFrame()


def calculate_team_rolling_stats(match_stats: pd.DataFrame, window: int = 10) -> Dict[str, pd.DataFrame]:
    """Calculate rolling averages for each team."""
    # Ensure date is datetime
    if 'date' in match_stats.columns:
        match_stats['date'] = pd.to_datetime(match_stats['date'])

    # Sort by date
    match_stats = match_stats.sort_values('date').reset_index(drop=True)

    # Get unique teams
    home_teams = set(match_stats['home_team'].dropna().unique())
    away_teams = set(match_stats['away_team'].dropna().unique())
    all_teams = home_teams | away_teams

    print(f"Processing {len(all_teams)} teams...")

    # Calculate rolling stats for each team
    team_stats = {}

    for team in all_teams:
        # Get all matches for this team (home or away)
        home_matches = match_stats[match_stats['home_team'] == team].copy()
        away_matches = match_stats[match_stats['away_team'] == team].copy()

        # Combine and rename columns
        home_matches = home_matches.rename(columns={
            'home_corners': 'corners_for',
            'away_corners': 'corners_against',
            'home_shots': 'shots_for',
            'away_shots': 'shots_against',
            'home_fouls': 'fouls_committed',
            'away_fouls': 'fouls_drawn',
        })
        home_matches['venue'] = 'home'

        away_matches = away_matches.rename(columns={
            'away_corners': 'corners_for',
            'home_corners': 'corners_against',
            'away_shots': 'shots_for',
            'home_shots': 'shots_against',
            'away_fouls': 'fouls_committed',
            'home_fouls': 'fouls_drawn',
        })
        away_matches['venue'] = 'away'

        # Combine
        team_matches = pd.concat([
            home_matches[['fixture_id', 'date', 'corners_for', 'corners_against',
                         'shots_for', 'shots_against', 'fouls_committed', 'fouls_drawn', 'venue']],
            away_matches[['fixture_id', 'date', 'corners_for', 'corners_against',
                         'shots_for', 'shots_against', 'fouls_committed', 'fouls_drawn', 'venue']]
        ]).sort_values('date').reset_index(drop=True)

        if len(team_matches) < 3:
            continue

        # Calculate rolling averages (excluding current match)
        for col in ['corners_for', 'corners_against', 'shots_for', 'shots_against',
                   'fouls_committed', 'fouls_drawn']:
            if col in team_matches.columns:
                team_matches[f'{col}_ema'] = (
                    team_matches[col]
                    .ewm(span=window, min_periods=3)
                    .mean()
                    .shift(1)  # Exclude current match
                )

        team_stats[team] = team_matches

    return team_stats


def add_corner_features_to_features(features_df: pd.DataFrame, team_stats: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add corner/shots features to the main features file."""
    print("\nAdding corner/shots features to features file...")

    # Create lookup for each fixture
    fixture_features = {}

    for team, stats in team_stats.items():
        for _, row in stats.iterrows():
            fixture_id = row['fixture_id']
            venue = row['venue']

            if fixture_id not in fixture_features:
                fixture_features[fixture_id] = {}

            prefix = 'home' if venue == 'home' else 'away'

            for col in ['corners_for_ema', 'corners_against_ema', 'shots_for_ema',
                       'shots_against_ema', 'fouls_committed_ema', 'fouls_drawn_ema']:
                if col in row and pd.notna(row[col]):
                    fixture_features[fixture_id][f'{prefix}_{col}'] = row[col]

    print(f"Created features for {len(fixture_features)} fixtures")

    # Add features to dataframe
    new_cols = [
        'home_corners_for_ema', 'home_corners_against_ema',
        'away_corners_for_ema', 'away_corners_against_ema',
        'home_shots_for_ema', 'home_shots_against_ema',
        'away_shots_for_ema', 'away_shots_against_ema',
        'home_fouls_committed_ema', 'home_fouls_drawn_ema',
        'away_fouls_committed_ema', 'away_fouls_drawn_ema',
    ]

    # Remove existing columns if present
    existing = [c for c in new_cols if c in features_df.columns]
    if existing:
        features_df = features_df.drop(columns=existing)

    # Add new columns
    for col in new_cols:
        features_df[col] = features_df['fixture_id'].map(
            lambda fid: fixture_features.get(fid, {}).get(col, np.nan)
        )

    # Create derived features
    features_df['expected_corners_total'] = (
        features_df['home_corners_for_ema'].fillna(5) +
        features_df['away_corners_for_ema'].fillna(4.5)
    )
    features_df['expected_shots_total'] = (
        features_df['home_shots_for_ema'].fillna(12) +
        features_df['away_shots_for_ema'].fillna(10)
    )
    features_df['expected_fouls_total'] = (
        features_df['home_fouls_committed_ema'].fillna(11) +
        features_df['away_fouls_committed_ema'].fillna(12)
    )

    # Cross-market features based on xgbfir findings
    # Shots predict corners
    features_df['cross_shots_corners'] = (
        features_df['home_shots_for_ema'].fillna(12) *
        features_df['away_shots_for_ema'].fillna(10)
    )
    # Corners predict shots
    features_df['cross_corners_shots'] = (
        features_df['home_corners_for_ema'].fillna(5) *
        features_df['away_corners_for_ema'].fillna(4.5)
    )

    return features_df


def main():
    print("=" * 70)
    print("REGENERATING FEATURES WITH CORNER/SHOTS EMAs")
    print("=" * 70)

    # Load match stats
    print("\n[1/3] Loading match_stats...")
    match_stats = load_all_match_stats()

    if match_stats.empty:
        print("No match_stats found!")
        return

    # Check columns
    print(f"\nMatch stats columns: {match_stats.columns.tolist()[:15]}...")

    # Calculate team rolling stats
    print("\n[2/3] Calculating team rolling stats...")
    team_stats = calculate_team_rolling_stats(match_stats)
    print(f"Processed {len(team_stats)} teams")

    # Load features file
    print("\n[3/3] Adding features to main file...")
    features_path = Path("data/03-features/features_all_5leagues_with_odds.csv")
    features_df = pd.read_csv(features_path)
    print(f"Loaded {len(features_df)} feature rows")

    # Add corner features
    features_df = add_corner_features_to_features(features_df, team_stats)

    # Check coverage
    corner_cols = [c for c in features_df.columns if 'corner' in c.lower()]
    print(f"\nCorner columns added: {len(corner_cols)}")
    for col in corner_cols[:5]:
        coverage = features_df[col].notna().sum()
        print(f"  {col}: {coverage}/{len(features_df)} ({coverage/len(features_df)*100:.1f}%)")

    # Save
    features_df.to_csv(features_path, index=False)
    print(f"\nSaved updated features to: {features_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total features: {len(features_df.columns)}")
    print(f"New corner/shots features: {len(corner_cols)}")


if __name__ == "__main__":
    main()
