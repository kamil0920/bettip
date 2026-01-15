"""Utility functions for paper trading."""
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_main_features(
    paths: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load the main features file.

    Args:
        paths: List of paths to try (in order). Defaults to common feature file locations.

    Returns:
        DataFrame with features

    Raises:
        FileNotFoundError: If no feature file found
    """
    if paths is None:
        paths = [
            'data/03-features/features_with_real_xg.csv',
            'data/03-features/features_all_leagues_complete.csv',
            'data/03-features/features_all_5leagues_with_odds.csv',
            'data/03-features/features_all.csv',
        ]

    for path in paths:
        filepath = Path(path)
        if filepath.exists():
            print(f"  Loading features from: {filepath.name}")
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            print(f"  Loaded {len(df)} matches")
            return df

    raise FileNotFoundError(f"No feature files found. Tried: {paths}")


def load_upcoming_fixtures(
    leagues: Optional[List[str]] = None,
    season: str = "2025",
    days_ahead: int = 7,
) -> pd.DataFrame:
    """
    Load upcoming fixtures from raw data.

    Args:
        leagues: List of leagues to load. Defaults to top 5 European leagues.
        season: Season folder name
        days_ahead: Number of days ahead to include

    Returns:
        DataFrame with upcoming fixtures
    """
    if leagues is None:
        leagues = ['premier_league', 'la_liga', 'serie_a', 'bundesliga', 'ligue_1']

    upcoming = []

    for league in leagues:
        matches_file = Path(f'data/01-raw/{league}/{season}/matches.parquet')
        if not matches_file.exists():
            continue

        try:
            df = pd.read_parquet(matches_file)
            df['league'] = league

            # Filter to not finished matches
            not_finished = df[df['fixture.status.short'] != 'FT'].copy()

            not_finished = not_finished.rename(columns={
                'fixture.id': 'fixture_id',
                'fixture.date': 'date',
                'teams.home.name': 'home_team',
                'teams.away.name': 'away_team',
                'fixture.referee': 'referee',
            })

            upcoming.append(not_finished)

        except Exception as e:
            print(f"  Warning: Could not load {league}: {e}")

    if not upcoming:
        print("No upcoming matches found")
        return pd.DataFrame()

    upcoming_df = pd.concat(upcoming, ignore_index=True)
    upcoming_df['date'] = pd.to_datetime(upcoming_df['date']).dt.tz_localize(None)
    upcoming_df = upcoming_df.sort_values('date')

    # Filter to next N days
    today = datetime.now()
    cutoff = today + timedelta(days=days_ahead)
    upcoming_df = upcoming_df[
        (upcoming_df['date'] >= today) &
        (upcoming_df['date'] <= cutoff)
    ]

    print(f"  Found {len(upcoming_df)} upcoming matches in next {days_ahead} days")
    return upcoming_df


def load_match_stats(
    leagues: Optional[List[str]] = None,
    season: str = "2025",
) -> pd.DataFrame:
    """
    Load match statistics from parquet files.

    Args:
        leagues: List of leagues to load
        season: Season folder name

    Returns:
        DataFrame with match stats
    """
    if leagues is None:
        leagues = ['premier_league', 'la_liga', 'serie_a']

    all_stats = []

    for league in leagues:
        stats_file = Path(f'data/01-raw/{league}/{season}/match_stats.parquet')
        matches_file = Path(f'data/01-raw/{league}/{season}/matches.parquet')

        if not stats_file.exists() or not matches_file.exists():
            continue

        try:
            stats = pd.read_parquet(stats_file)
            matches = pd.read_parquet(matches_file)

            # Get referee info from matches
            matches_slim = matches[[
                'fixture.id', 'fixture.referee'
            ]].rename(columns={
                'fixture.id': 'fixture_id',
                'fixture.referee': 'referee',
            })

            merged = stats.merge(matches_slim, on='fixture_id', how='left')
            merged['league'] = league
            merged['season'] = season

            # Calculate totals
            if 'home_corners' in merged.columns:
                merged['total_corners'] = merged['home_corners'] + merged['away_corners']
            if 'home_shots' in merged.columns:
                merged['total_shots'] = merged['home_shots'] + merged['away_shots']
            if 'home_fouls' in merged.columns:
                merged['total_fouls'] = merged['home_fouls'] + merged['away_fouls']

            all_stats.append(merged)

        except Exception as e:
            print(f"  Warning: Could not load {league} stats: {e}")

    if not all_stats:
        return pd.DataFrame()

    return pd.concat(all_stats, ignore_index=True)


def calculate_referee_stats_from_train(
    train_df: pd.DataFrame,
    stat_column: str,
) -> pd.DataFrame:
    """
    Calculate referee statistics from training data only.

    This prevents data leakage by only using historical data.

    Args:
        train_df: Training DataFrame with referee and stat columns
        stat_column: Column name for the stat (e.g., 'total_corners')

    Returns:
        DataFrame with referee stats (ref_avg, ref_std, ref_matches)
    """
    if 'referee' not in train_df.columns or stat_column not in train_df.columns:
        return pd.DataFrame()

    referee_stats = train_df.groupby('referee').agg({
        stat_column: ['mean', 'std', 'count']
    }).reset_index()

    referee_stats.columns = ['referee', 'ref_avg', 'ref_std', 'ref_matches']

    # Fill NaN std with overall std
    overall_std = train_df[stat_column].std()
    referee_stats['ref_std'] = referee_stats['ref_std'].fillna(overall_std)

    return referee_stats


def apply_referee_stats(
    df: pd.DataFrame,
    referee_stats: pd.DataFrame,
    default_avg: float,
    default_std: float,
) -> pd.DataFrame:
    """
    Apply referee statistics to a DataFrame.

    Args:
        df: DataFrame to add referee stats to
        referee_stats: DataFrame with referee statistics
        default_avg: Default average for unknown referees
        default_std: Default std for unknown referees

    Returns:
        DataFrame with ref_avg and ref_std columns added
    """
    if referee_stats.empty:
        df['ref_avg'] = default_avg
        df['ref_std'] = default_std
        return df

    df = df.merge(referee_stats, on='referee', how='left')
    df['ref_avg'] = df['ref_avg'].fillna(default_avg)
    df['ref_std'] = df['ref_std'].fillna(default_std)

    return df
