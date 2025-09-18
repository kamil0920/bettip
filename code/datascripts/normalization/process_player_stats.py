import pandas as pd
import numpy as np

def process_player_stats(player_stats: pd.DataFrame, drop_old_features=True) -> pd.DataFrame:
    df = player_stats.copy()

    # Ensure unified datetime for ordering
    if 'fixture_dt' not in df.columns and 'fixture_date' in df.columns:
        df['fixture_dt'] = pd.to_datetime(df['fixture_date'], errors='coerce')
    else:
        df['fixture_dt'] = pd.to_datetime(df.get('fixture_dt'), utc=True, errors='coerce')

    # Rating source if only games_rating exists
    if 'rating' not in df.columns and 'games_rating' in df.columns:
        df['rating'] = pd.to_numeric(df['games_rating'], errors='coerce')

    # Per-90 inputs
    df['minutes'] = pd.to_numeric(df.get('minutes'), errors='coerce').fillna(0)
    for col in ['shots_on','shots_total','passes_key']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[f'{col}_per90'] = np.where(df['minutes'] > 0, (df[col]/df['minutes']) * 90.0, 0.0)

    df['minutes_share'] = np.where(df['minutes'] > 0, df['minutes'] / 90.0, 0.0)

    # Order → shift → EMA
    df = df.sort_values(['player_id','fixture_dt']).copy()
    base_cols = [c for c in ['rating','shots_on_per90','shots_total_per90','passes_key_per90','minutes_share'] if c in df.columns]

    for c in base_cols:
        df[f'{c}_shifted'] = df.groupby('player_id', group_keys=False)[c].shift(1)

    for c in base_cols:
        df[f'{c}_ema5'] = (
            df.groupby('player_id', group_keys=False)[f'{c}_shifted']
              .apply(lambda s: s.ewm(span=5, adjust=False).mean())
        )  # EMA per pandas docs [4][5]

    df.drop(columns=[f'{c}_shifted' for c in base_cols], inplace=True, errors='ignore')

    if drop_old_features:
        keep = [
            'fixture_id','player_id','team_id','fixture_dt',
            'rating_ema5','shots_on_per90_ema5','shots_total_per90_ema5',
            'passes_key_per90_ema5','minutes_share_ema5'
        ]
        df = df[[c for c in keep if c in df.columns]].copy()

    ema_cols = [c for c in df.columns if c.endswith('_ema5')]
    if ema_cols:
        df[ema_cols] = df[ema_cols].fillna(0)

    return df

# Example Usage
# Assuming player_stats is your DataFrame from parquet
# processed_df = process_player_stats(player_stats, drop_old_features=True)
# processed_df.to_parquet('processed_player_stats.parquet')

def add_fixture_datetime(player_stats: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    m = matches[['fixture_id','date','timestamp']].copy()
    out = player_stats.merge(m, on='fixture_id', how='left')
    dt_from_str = pd.to_datetime(out['date'], utc=True, errors='coerce')
    dt_from_ts  = pd.to_datetime(out['timestamp'], unit='s', utc=True, errors='coerce')
    out['fixture_dt'] = dt_from_str.fillna(dt_from_ts)
    return out


def add_per90(df: pd.DataFrame, cols=('shots_on','shots_total','passes_key','goals','assists')) -> pd.DataFrame:
    df = df.copy()
    # Ensure minutes is numeric and non-null for math
    df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0)

    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            df[f'{c}_per90'] = np.where(df['minutes'] > 0, (df[c] / df['minutes']) * 90.0, 0.0)

    # Availability proxy
    df['minutes_share'] = np.where(df['minutes'] > 0, df['minutes'] / 90.0, 0.0)
    return df

def add_player_ema(df: pd.DataFrame, span=5) -> pd.DataFrame:
    df = df.sort_values(['player_id','fixture_dt']).copy()

    base_cols = []
    if 'games_rating' in df.columns:             # rename to 'rating' for convenience
        df['rating'] = pd.to_numeric(df['games_rating'], errors='coerce')
        base_cols.append('rating')

    # Include the per-90 inputs just created if present
    for c in ['shots_on_per90','shots_total_per90','passes_key_per90','minutes_share']:
        if c in df.columns:
            base_cols.append(c)

    # Shift to use only past information
    for c in base_cols:
        df[f'{c}_shifted'] = df.groupby('player_id', group_keys=False)[c].shift(1)

    # EMA per player (span controls decay speed)
    for c in base_cols:
        ema_col = f'{c}_ema5'
        df[ema_col] = (
            df
            .groupby('player_id', group_keys=False)[f'{c}_shifted']
            .apply(lambda s: s.ewm(span=span, adjust=False).mean())
        )

    # Clean up temporary shifted columns
    df.drop(columns=[f'{c}_shifted' for c in base_cols], inplace=True, errors='ignore')
    return df

def build_player_form_features(player_stats: pd.DataFrame, matches: pd.DataFrame,
                               keep_lead_cols=True, drop_old_features=True) -> pd.DataFrame:
    df = add_fixture_datetime(player_stats, matches)
    df = add_per90(df, cols=('shots_on','shots_total','passes_key','goals','assists'))

    df = add_player_ema(df, span=5)

    mask_played = (pd.to_numeric(df.get('minutes'), errors='coerce').fillna(0) > 0) & (df.get('games_rating').notna())
    df = df.loc[mask_played].copy()

    # Lean set to keep (customize as needed)
    keep = [
        'fixture_id','player_id','team_id','fixture_dt',
        'rating_ema5','shots_on_per90_ema5','shots_total_per90_ema5',
        'passes_key_per90_ema5','minutes_share_ema5'
    ]
    # Preserve label columns if provided upstream
    if keep_lead_cols:
        keep += [c for c in ('games_rating','minutes') if c in df.columns]

    out = df[[c for c in keep if c in df.columns]].copy()
    return out
