import pandas as pd
import numpy as np


def process_player_stats(player_stats: pd.DataFrame, drop_old_features=True) -> pd.DataFrame:
    """
    Process player stats: Normalize to per-90, sort, shift, compute EMA, and keep lean features.

    Args:
    player_stats: DataFrame with columns like 'player_id', 'fixture_date', 'rating', 'shots_on', etc.
    drop_old_features: If True, drop raw columns after creating EMAs.

    Returns:
    Processed DataFrame with EMA features.
    """
    df = player_stats.copy()

    # Step 1: Compute per-90 rates (handle division by zero)
    per90_cols = ['shots_on', 'shots_total', 'passes_key']  # Add more (e.g., 'xg', 'xa')
    for col in per90_cols:
        if col in df.columns:
            df[f'{col}_per90'] = np.where(
                df['minutes'] > 0,
                (df[col] / df['minutes']) * 90,
                0
            )

    # Add minutes_share (e.g., fraction of full game; assume max 90, customize if extra time)
    df['minutes_share'] = np.where(
        df['minutes'] > 0,
        df['minutes'] / 90,
        0
    )

    # Step 2: Sort by player and date
    df = df.sort_values(['player_id', 'fixture_date'])

    # Step 3: Shift stats by 1 to avoid leakage
    shift_cols = ['rating', 'minutes_share'] + [f'{col}_per90' for col in per90_cols]
    for col in shift_cols:
        if col in df.columns:
            df[f'{col}_shifted'] = df.groupby('player_id')[col].shift(1)

    # Step 4: Compute EMA on shifted columns (span=5, adjust=False for simple EMA)
    ema_cols = [f'{col}_shifted' for col in shift_cols]
    for col in ema_cols:
        ema_col = col.replace('_shifted', '_ema5')
        df[ema_col] = (
            df
            .groupby('player_id', group_keys=False)[col]
            .apply(lambda s: s.ewm(span=5, adjust=False).mean())
        )

    # Step 5: Drop intermediates and old features if requested
    df = df.drop(columns=ema_cols)  # Drop shifted temps
    if drop_old_features:
        # Keep lean set: IDs, date, EMAs (customize)
        keep_cols = [
            'player_id', 'fixture_date', 'team_id',  # Essentials
            'rating_ema5', 'shots_on_per90_ema5', 'shots_total_per90_ema5',
            'passes_key_per90_ema5', 'minutes_share_ema5'
        ]
        df = df[[col for col in keep_cols if col in df.columns]]

    # Fill any NaN EMAs (e.g., first match) with 0 or column mean
    ema_fill_cols = [col for col in df.columns if '_ema5' in col]
    df[ema_fill_cols] = df[ema_fill_cols].fillna(0)

    return df

# Example Usage
# Assuming player_stats is your DataFrame from parquet
# processed_df = process_player_stats(player_stats, drop_old_features=True)
# processed_df.to_parquet('processed_player_stats.parquet')
