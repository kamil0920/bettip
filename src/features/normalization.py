"""
Rolling Z-Score Normalization for Distribution Shift Mitigation.

This module applies per-league rolling z-score normalization to numeric features
to neutralize temporal distribution drift. The adversarial classifier can trivially
distinguish early from late data when features represent ABSOLUTE values that drift
over time (e.g., league-average fouls increased from 22 to 25 per match over 6 years).

Rolling z-score changes cross-temporal rank ordering (unlike global StandardScaler):
- Match A (2019): ref_cards_avg=5.0, local avg=4.0 → z=2.0
- Match B (2025): ref_cards_avg=5.5, local avg=5.0 → z=1.0
- Raw: B > A. Z-scored: A > B. **Rank reversed** → trees can't distinguish time periods.

Usage:
    from src.features.normalization import apply_rolling_zscore
    df = apply_rolling_zscore(df, league_col='league', date_col='date')
"""
import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Structural / ID columns — never normalize
_STRUCTURAL_COLS = {
    'fixture_id', 'date', 'home_team_id', 'home_team_name',
    'away_team_id', 'away_team_name', 'round', 'fixture_round',
    'season', 'league', 'league_id',
}

# Target / outcome columns — never normalize
_TARGET_COLS = {
    'home_win', 'draw', 'away_win', 'match_result', 'result',
    'btts', 'over25', 'under25',
    'total_goals', 'goal_difference', 'home_goals', 'away_goals',
    'ft_home', 'ft_away', 'gd_form_diff',
}

# Raw match stats (post-match leakage) — never normalize
_LEAKAGE_COLS = {
    'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
    'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
    'home_cards', 'away_cards', 'home_possession', 'away_possession',
    'home_offsides', 'away_offsides',
    'home_yellows', 'away_yellows', 'home_reds', 'away_reds',
    'total_corners', 'total_fouls', 'total_shots', 'total_cards',
    'total_yellows', 'total_reds', 'total_shots_on_target',
    'home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards',
}

# Self-normalizing columns — ELO is a rating system, odds are market-driven
_SELF_NORMALIZING_PREFIXES = (
    'elo_', 'odds_', 'theodds_',
)
_SELF_NORMALIZING_EXACT = {
    'home_elo', 'away_elo', 'elo_diff',
    'home_win_prob_elo', 'away_win_prob_elo',
    'home_venue_elo', 'away_venue_elo', 'venue_elo_diff',
    'home_team_venue_gap', 'away_team_venue_gap',
    'overround', 'overround_change',
}

# Already-relative columns — percentages, rates, biases, Bayesian posteriors
_RELATIVE_SUFFIXES = (
    '_bias', '_pct', '_rate',
)
_RELATIVE_PREFIXES = (
    'bayes_', 'oa_',
    'home_bayes_', 'away_bayes_',
    'home_oa_', 'away_oa_',
)

# Binary flag prefixes
_BINARY_PREFIXES = (
    'is_', 'weather_is_',
)

# Bounded count columns — already bounded by small N
_BOUNDED_SUFFIXES = (
    '_wins_last_n', '_draws_last_n', '_losses_last_n',
)
_BOUNDED_PREFIXES = (
    'h2h_',
)
_BOUNDED_EXACT = {
    'ref_matches', 'home_rest_days', 'away_rest_days', 'rest_advantage',
}

# Probability columns (already 0-1 bounded)
_PROBABILITY_SUFFIXES = (
    '_prob', '_win_prob',
)
_PROBABILITY_EXACT = {
    'poisson_home_win_prob', 'poisson_draw_prob', 'poisson_away_win_prob',
    'glm_home_win_prob', 'glm_draw_prob', 'glm_away_win_prob',
    'sharp_confidence',
}


def get_columns_to_normalize(df: pd.DataFrame) -> List[str]:
    """
    Determine which columns should receive rolling z-score normalization.

    Excludes:
    - Structural/ID columns (fixture_id, date, team IDs, etc.)
    - Target/outcome columns (home_win, btts, total_goals, etc.)
    - Leakage columns (raw match stats)
    - Self-normalizing columns (ELO ratings, odds)
    - Already-relative columns (*_bias, *_pct, *_rate, bayes_*, oa_*)
    - Binary columns (is_*, weather_is_*, or columns with only {0, 1} values)
    - Bounded counts (*_wins_last_n, h2h_*, ref_matches)
    - Probability columns (*_prob, already 0-1)

    Args:
        df: DataFrame with feature columns

    Returns:
        List of column names to normalize
    """
    cols_to_normalize = []

    for col in df.columns:
        # Skip non-numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Skip structural
        if col in _STRUCTURAL_COLS:
            continue

        # Skip targets
        if col in _TARGET_COLS:
            continue

        # Skip leakage
        if col in _LEAKAGE_COLS:
            continue

        # Skip self-normalizing (exact match)
        if col in _SELF_NORMALIZING_EXACT:
            continue

        # Skip self-normalizing (prefix match)
        if col.startswith(_SELF_NORMALIZING_PREFIXES):
            continue

        # Skip already-relative (suffix match)
        if any(col.endswith(s) for s in _RELATIVE_SUFFIXES):
            continue

        # Skip already-relative (prefix match)
        if col.startswith(_RELATIVE_PREFIXES):
            continue

        # Skip binary flags by prefix
        if col.startswith(_BINARY_PREFIXES):
            continue

        # Skip bounded counts (suffix)
        if any(col.endswith(s) for s in _BOUNDED_SUFFIXES):
            continue

        # Skip bounded counts (prefix)
        if col.startswith(_BOUNDED_PREFIXES):
            continue

        # Skip bounded counts (exact)
        if col in _BOUNDED_EXACT:
            continue

        # Skip probability columns (suffix)
        if any(col.endswith(s) for s in _PROBABILITY_SUFFIXES):
            continue

        # Skip probability columns (exact)
        if col in _PROBABILITY_EXACT:
            continue

        # Skip binary columns (values are only 0 and 1)
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            continue

        cols_to_normalize.append(col)

    return cols_to_normalize


def apply_rolling_zscore(
    df: pd.DataFrame,
    league_col: str = 'league',
    date_col: str = 'date',
    min_periods: int = 30,
    window: int = 0,
) -> pd.DataFrame:
    """
    Apply per-league rolling z-score normalization to numeric features.

    For each normalizable column, grouped by league:
    - shifted = col.shift(1)  (avoid look-ahead bias)
    - rolling_mean = shifted.expanding(min_periods).mean()  (or rolling(window))
    - rolling_std = shifted.expanding(min_periods).std()
    - normalized = (col - rolling_mean) / rolling_std
    - Fill NaN (early rows or zero std) with 0.0 (neutral z-score)

    Features are REPLACED in-place (not added alongside originals).

    Args:
        df: DataFrame sorted by date with feature columns
        league_col: Column name for league grouping. Falls back to 'league_id',
                    then global normalization if neither exists.
        date_col: Column name for date/time ordering
        min_periods: Minimum observations before computing stats (default: 30)
        window: Rolling window size. 0 means expanding (use all history).

    Returns:
        DataFrame with normalized features replacing originals
    """
    df = df.copy()

    # Resolve league column
    actual_league_col = None
    if league_col in df.columns:
        actual_league_col = league_col
    elif 'league_id' in df.columns:
        actual_league_col = 'league_id'
        logger.info(f"League column '{league_col}' not found, falling back to 'league_id'")
    else:
        logger.warning(f"No league column found (tried '{league_col}', 'league_id'). "
                       "Using global normalization.")

    # Get columns to normalize
    cols = get_columns_to_normalize(df)
    if not cols:
        logger.warning("No columns selected for normalization")
        return df

    logger.info(f"Normalizing {len(cols)} columns with "
                f"{'expanding' if window == 0 else f'rolling({window})'} window, "
                f"min_periods={min_periods}")

    # Sort by league + date for correct temporal ordering
    sort_cols = [actual_league_col, date_col] if actual_league_col else [date_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if actual_league_col:
        # Per-league normalization
        normalized_chunks = []
        for league_val, group in df.groupby(actual_league_col, sort=False):
            group = _normalize_group(group, cols, min_periods, window)
            normalized_chunks.append(group)
        df = pd.concat(normalized_chunks, ignore_index=True)
    else:
        # Global normalization
        df = _normalize_group(df, cols, min_periods, window)

    # Re-sort by date (restore chronological order across all leagues)
    df = df.sort_values(date_col).reset_index(drop=True)

    return df


def _normalize_group(
    group: pd.DataFrame,
    cols: List[str],
    min_periods: int,
    window: int,
) -> pd.DataFrame:
    """
    Apply rolling z-score to a single group (league or global).

    Uses shift(1) to avoid look-ahead bias: the z-score at time T
    uses only statistics from T-1 and earlier.
    """
    group = group.copy()

    for col in cols:
        if col not in group.columns:
            continue

        values = group[col].astype(float)

        # Shift to avoid look-ahead: statistics at row i use rows 0..i-1
        shifted = values.shift(1)

        if window > 0:
            rolling_mean = shifted.rolling(window=window, min_periods=min_periods).mean()
            rolling_std = shifted.rolling(window=window, min_periods=min_periods).std()
        else:
            rolling_mean = shifted.expanding(min_periods=min_periods).mean()
            rolling_std = shifted.expanding(min_periods=min_periods).std()

        # Compute z-score; std=0 produces NaN which gets filled below
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = (values - rolling_mean) / rolling_std

        # Replace inf/-inf and NaN with 0.0 (neutral z-score)
        normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        group[col] = normalized

    return group
