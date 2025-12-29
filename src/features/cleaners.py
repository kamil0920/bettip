"""Data cleaning utilities for feature engineering."""
import numpy as np
import pandas as pd

from src.features.interfaces import IDataCleaner


class BasicDataCleaner(IDataCleaner):
    """Basic data cleaner."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic clean: duplicates, sorting.

        Args:
            df: DataFrame

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        if removed > 0:
            print(f"  Removed {removed} duplicates")

        return df_clean


class MatchDataCleaner(IDataCleaner):
    """Match data cleaner with column mapping for raw API format."""

    COLUMN_MAPPING = {
        'fixture.id': 'fixture_id',
        'fixture.date': 'date',
        'fixture.referee': 'referee',
        'fixture.venue.id': 'venue_id',
        'fixture.venue.name': 'venue_name',
        'fixture.status.short': 'status',
        'league.id': 'league_id',
        'league.round': 'round',
        'league.season': 'season',
        'teams.home.id': 'home_team_id',
        'teams.home.name': 'home_team_name',
        'teams.away.id': 'away_team_id',
        'teams.away.name': 'away_team_name',
        'goals.home': 'ft_home',
        'goals.away': 'ft_away',
        'score.halftime.home': 'ht_home',
        'score.halftime.away': 'ht_away',
    }

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean match data with automatic column mapping.

        Args:
            df: DataFrame (raw API format or already cleaned)

        Returns:
            Cleaned DataFrame with standardized column names
        """
        df_clean = df.copy()

        if 'fixture.id' in df_clean.columns:
            df_clean = self._apply_column_mapping(df_clean)

        if 'home_team_name' in df_clean.columns and 'home_team' not in df_clean.columns:
            df_clean['home_team'] = df_clean['home_team_name']
        if 'away_team_name' in df_clean.columns and 'away_team' not in df_clean.columns:
            df_clean['away_team'] = df_clean['away_team_name']

        score_cols = self._get_score_columns(df_clean)
        if score_cols:
            df_clean = df_clean.dropna(subset=score_cols)

        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'])
            df_clean = df_clean.sort_values('date').reset_index(drop=True)

        print(f"Matches: {len(df_clean)} (with full scores)")
        return df_clean

    def _apply_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column mapping from raw API format to clean format."""
        rename_map = {}
        for raw_col, clean_col in self.COLUMN_MAPPING.items():
            if raw_col in df.columns and clean_col not in df.columns:
                rename_map[raw_col] = clean_col

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _get_score_columns(self, df: pd.DataFrame) -> list:
        """Get available score columns for filtering."""
        if 'ft_home' in df.columns and 'ft_away' in df.columns:
            return ['ft_home', 'ft_away']
        elif 'goals.home' in df.columns and 'goals.away' in df.columns:
            return ['goals.home', 'goals.away']
        return []


class PlayerStatsDataCleaner(IDataCleaner):
    """Player stats data cleaner."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean players stats data.

        Args:
            df: DataFrame

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        if df_clean.empty:
            print(f"Player stats: 0 records (empty)")
            return df_clean

        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(0)

        minutes_col = None
        for col_name in ['minutes', 'games.minutes']:
            if col_name in df_clean.columns:
                minutes_col = col_name
                break

        if minutes_col:
            df_clean = df_clean[df_clean[minutes_col] > 0]
            print(f"Player stats: {len(df_clean)} records (filtered by {minutes_col})")
        else:
            # No minutes column - data might be in nested format, skip filtering
            print(f"Player stats: {len(df_clean)} records (no minutes column found)")

        return df_clean
