import numpy as np
import pandas as pd
from interfaces import IDataCleaner


class BasicDataCleaner(IDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic clean: duplicates, sorting

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

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean march data

        Args:
            df: DataFrame

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=['ft_home', 'ft_away'])
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        print(f"  Matches: {len(df_clean)} (with full scores)")

        return df_clean


class PlayerStatsDataCleaner(IDataCleaner):

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean players stats data

        Args:
            df: DataFrame

        Returns:
            Cleanded DataFrame
        """
        df_clean = df.copy()

        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = df_clean[numeric_columns].fillna(0)

        df_clean = df_clean[df_clean['minutes'] > 0]

        print(f"  Player stats: {len(df_clean)} records")

        return df_clean