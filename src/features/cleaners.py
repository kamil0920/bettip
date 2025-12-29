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
    """Match data cleaner."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean match data.

        Args:
            df: DataFrame

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=['ft_home', 'ft_away'])
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        print(f"Matches: {len(df_clean)} (with full scores)")

        return df_clean


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
