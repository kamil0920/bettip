"""Shared utilities for match statistics normalization."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_match_stats_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename API-Football column names to pipeline standard names.

    API returns: corner_kicks, total_shots, shots_on_goal, ball_possession
    Pipeline expects: corners, shots, shots_on_target, possession

    Some leagues (Bundesliga, Ligue 1) have older parquet files with the
    API names.  Applying this after every ``pd.read_parquet()`` call
    ensures consistent column names regardless of when the data was
    collected.
    """
    col_renames = {}
    for col in df.columns:
        renamed = col
        renamed = renamed.replace('corner_kicks', 'corners')
        renamed = renamed.replace('total_shots', 'shots')
        renamed = renamed.replace('shots_on_goal', 'shots_on_target')
        renamed = renamed.replace('ball_possession', 'possession')
        if renamed != col:
            col_renames[col] = renamed
    if col_renames:
        df = df.rename(columns=col_renames)
        logger.info(f"Normalized match_stats columns: {col_renames}")
    return df
