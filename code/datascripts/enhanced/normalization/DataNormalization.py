import numpy as np
import pandas as pd


class DataNormalization:
    """
    Cleans and normalizes the raw DataFrame:
      - Parses nested JSON fields if any remain.
      - Standardizes column names/schema.
      - Handles missing values and type conversions.
    """

    def __init__(self, required_columns):
        self.required_columns = required_columns

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure required columns exist; fill missing columns with NaN
        for col in self.required_columns:
            if col not in df.columns:
                df[col] = np.nan

        # Example: parse nested JSON in a cell (if any)
        # e.g., df['player'] might be dict; flatten or extract:
        if 'player' in df.columns and df['player'].dtype == object:
            # Expand dicts into columns (assuming structure {'id':..,'name':..})
            player_df = pd.json_normalize(df['player']).add_prefix('player_')
            df = pd.concat([df.drop(columns=['player']), player_df], axis=1)

        # Handle missing values: numeric -> 0 or median, categorical -> 'Unknown'
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col] = df[col].fillna(0)
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna('Unknown')

        # Enforce types (e.g., minutes as int)
        if 'minutes_played' in df.columns:
            df['minutes_played'] = df['minutes_played'].fillna(0).astype(int)

        # Final check: ensure schema matches desired columns
        df = df[self.required_columns]
        return df