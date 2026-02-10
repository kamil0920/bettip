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

    When old-name and new-name columns both exist (e.g. after pd.concat
    of old + new season files), the old column is coalesced into the new
    one and dropped to prevent duplicate column names.
    """
    _RENAMES = [
        ('corner_kicks', 'corners'),
        ('total_shots', 'shots'),
        ('shots_on_goal', 'shots_on_target'),
        ('ball_possession', 'possession'),
    ]

    col_renames = {}
    for col in df.columns:
        renamed = col
        for old, new in _RENAMES:
            renamed = renamed.replace(old, new)
        if renamed != col:
            col_renames[col] = renamed

    if not col_renames:
        return df

    # Check for conflicts: old column would rename to a name that already
    # exists (e.g. concat of old-season home_corner_kicks + new-season
    # home_corners).  Coalesce old into new and drop the old column.
    for old_col, new_col in list(col_renames.items()):
        if new_col in df.columns:
            df[new_col] = df[new_col].fillna(df[old_col])
            df = df.drop(columns=[old_col])
            logger.debug(f"Coalesced duplicate column {old_col} into {new_col}")
            del col_renames[old_col]

    if col_renames:
        df = df.rename(columns=col_renames)
        logger.debug(f"Normalized match_stats columns: {col_renames}")

    return df
