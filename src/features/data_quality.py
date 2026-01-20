"""
Data quality validation and imputation for betting features.

This module provides:
1. NaN detection and reporting
2. Domain-specific imputation strategies
3. Data completeness flags for downstream callibration
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Summary of data quality for a dataset."""

    total_rows: int
    complete_rows: int
    completeness_ratio: float
    nan_by_column: Dict[str, float]
    critical_missing: List[str]
    imputed_columns: List[str]

    def __str__(self) -> str:
        return (
            f"DataQualityReport:\n"
            f"  Total rows: {self.total_rows}\n"
            f"  Complete rows: {self.complete_rows} ({self.completeness_ratio:.1%})\n"
            f"  Critical missing: {self.critical_missing}\n"
            f"  Imputed columns: {self.imputed_columns}"
        )


# Feature groups by data availability
CRITICAL_EMA_FEATURES = [
    'home_fouls_committed_ema', 'away_fouls_committed_ema',
    'home_shots_total_ema', 'away_shots_total_ema',
    'home_shots_on_ema', 'away_shots_on_ema',
    'home_rating_ema', 'away_rating_ema',
]

ALWAYS_AVAILABLE_FEATURES = [
    'home_goals_scored_ema', 'away_goals_scored_ema',
    'home_goals_conceded_ema', 'away_goals_conceded_ema',
    'home_points_ema', 'away_points_ema',
    'elo_home', 'elo_away', 'elo_diff',
    'home_avg_yellows', 'away_avg_yellows',
]

# League-specific medians for imputation (from historical data analysis)
LEAGUE_MEDIANS = {
    'serie_a': {
        'fouls_committed_ema': 13.5,
        'shots_total_ema': 12.0,
        'shots_on_ema': 4.5,
        'rating_ema': 6.8,
    },
    'premier_league': {
        'fouls_committed_ema': 10.5,
        'shots_total_ema': 13.0,
        'shots_on_ema': 4.8,
        'rating_ema': 6.9,
    },
    'bundesliga': {
        'fouls_committed_ema': 12.0,
        'shots_total_ema': 13.5,
        'shots_on_ema': 5.0,
        'rating_ema': 6.85,
    },
    'la_liga': {
        'fouls_committed_ema': 13.0,
        'shots_total_ema': 12.5,
        'shots_on_ema': 4.3,
        'rating_ema': 6.75,
    },
    'ligue_1': {
        'fouls_committed_ema': 14.0,
        'shots_total_ema': 12.0,
        'shots_on_ema': 4.2,
        'rating_ema': 6.7,
    },
    'default': {
        'fouls_committed_ema': 12.5,
        'shots_total_ema': 12.5,
        'shots_on_ema': 4.5,
        'rating_ema': 6.8,
    },
}


def analyze_nan_distribution(df: pd.DataFrame) -> DataQualityReport:
    """
    Analyze NaN distribution in a feature DataFrame.

    Args:
        df: Feature DataFrame to analyze

    Returns:
        DataQualityReport with detailed NaN statistics
    """
    total_rows = len(df)

    # Calculate NaN percentage per column
    nan_by_column = {}
    for col in df.columns:
        nan_pct = df[col].isna().mean()
        if nan_pct > 0:
            nan_by_column[col] = nan_pct

    # Identify critical missing features
    critical_missing = []
    for feat in CRITICAL_EMA_FEATURES:
        if feat in df.columns and df[feat].isna().mean() > 0.1:
            critical_missing.append(feat)

    # Count complete rows (no NaN in critical features)
    critical_cols = [c for c in CRITICAL_EMA_FEATURES if c in df.columns]
    if critical_cols:
        complete_mask = df[critical_cols].notna().all(axis=1)
        complete_rows = complete_mask.sum()
    else:
        complete_rows = total_rows

    completeness_ratio = complete_rows / total_rows if total_rows > 0 else 0

    return DataQualityReport(
        total_rows=total_rows,
        complete_rows=complete_rows,
        completeness_ratio=completeness_ratio,
        nan_by_column=nan_by_column,
        critical_missing=critical_missing,
        imputed_columns=[],
    )


def detect_league(df: pd.DataFrame, row_idx: int = None) -> str:
    """
    Detect league from DataFrame or row for imputation.

    Args:
        df: DataFrame with league information
        row_idx: Optional specific row to check

    Returns:
        League name or 'default'
    """
    league_cols = ['league', 'competition', 'league_name']

    for col in league_cols:
        if col in df.columns:
            if row_idx is not None:
                league_val = df.iloc[row_idx][col]
            else:
                league_val = df[col].mode().iloc[0] if len(df) > 0 else None

            if pd.notna(league_val):
                league_str = str(league_val).lower().replace(' ', '_')
                for known_league in LEAGUE_MEDIANS.keys():
                    if known_league in league_str or league_str in known_league:
                        return known_league

    return 'default'


def impute_with_league_medians(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    inplace: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Impute missing EMA features using league-specific medians.

    This is a fallback strategy when team-specific data is unavailable.
    Imputed values are less accurate but prevent model failures.

    Args:
        df: Feature DataFrame with potential NaN values
        columns: Specific columns to impute (default: CRITICAL_EMA_FEATURES)
        inplace: Whether to modify df in place

    Returns:
        Tuple of (imputed DataFrame, list of imputed column names)
    """
    if not inplace:
        df = df.copy()

    if columns is None:
        columns = CRITICAL_EMA_FEATURES

    imputed_cols = []

    for col in columns:
        if col not in df.columns:
            continue

        nan_mask = df[col].isna()
        if not nan_mask.any():
            continue

        # Extract base feature name (e.g., 'fouls_committed_ema' from 'home_fouls_committed_ema')
        base_name = col.replace('home_', '').replace('away_', '')

        # Impute row by row using league-specific medians
        for idx in df.index[nan_mask]:
            league = detect_league(df, idx)
            medians = LEAGUE_MEDIANS.get(league, LEAGUE_MEDIANS['default'])

            if base_name in medians:
                df.loc[idx, col] = medians[base_name]

        if nan_mask.any():
            imputed_cols.append(col)
            logger.info(f"Imputed {nan_mask.sum()} values in {col} using league medians")

    return df, imputed_cols


def add_data_quality_flags(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Add flags indicating data completeness for each row.

    These flags help downstream callibration know when predictions
    are based on imputed vs actual data.

    Args:
        df: Feature DataFrame
        inplace: Whether to modify df in place

    Returns:
        DataFrame with added quality flag columns
    """
    if not inplace:
        df = df.copy()

    # Check which critical features are present
    present_critical = [c for c in CRITICAL_EMA_FEATURES if c in df.columns]

    if present_critical:
        # Flag for complete EMA data
        df['_has_complete_ema'] = df[present_critical].notna().all(axis=1)

        # Count how many critical features are available
        df['_ema_completeness'] = df[present_critical].notna().sum(axis=1) / len(present_critical)
    else:
        df['_has_complete_ema'] = False
        df['_ema_completeness'] = 0.0

    # Flag for imputed data (set during imputation)
    if '_is_imputed' not in df.columns:
        df['_is_imputed'] = False

    return df


def validate_features_for_prediction(
    df: pd.DataFrame,
    required_features: List[str],
    allow_imputation: bool = True,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Validate and prepare features for model prediction.

    This is the main entry point for data quality handling.

    Args:
        df: Feature DataFrame for prediction
        required_features: List of features the model needs
        allow_imputation: Whether to impute missing values

    Returns:
        Tuple of (validated DataFrame, quality report)
    """
    # Initial analysis
    report = analyze_nan_distribution(df)
    logger.info(f"Initial data quality: {report.completeness_ratio:.1%} complete")

    # Check which required features are missing
    missing_required = [f for f in required_features if f not in df.columns]
    if missing_required:
        logger.warning(f"Missing required features (not in DataFrame): {missing_required}")

    # Add quality flags before imputation
    df = add_data_quality_flags(df)

    # Impute if allowed
    if allow_imputation:
        impute_cols = [f for f in required_features if f in CRITICAL_EMA_FEATURES]
        df, imputed = impute_with_league_medians(df, columns=impute_cols)

        # Mark imputed rows
        for col in imputed:
            # Rows that were NaN before imputation are now marked
            df.loc[df['_ema_completeness'] < 1.0, '_is_imputed'] = True

        report.imputed_columns = imputed

    # Final quality check
    final_report = analyze_nan_distribution(df)
    final_report.imputed_columns = report.imputed_columns

    logger.info(f"Final data quality: {final_report.completeness_ratio:.1%} complete")
    logger.info(f"Imputed columns: {final_report.imputed_columns}")

    return df, final_report


def get_prediction_confidence(
    row: pd.Series,
    base_confidence: float = 1.0,
) -> float:
    """
    Calculate prediction confidence based on data quality.

    Predictions made with imputed data should have lower confidence.

    Args:
        row: Single row from feature DataFrame
        base_confidence: Starting confidence level

    Returns:
        Adjusted confidence score (0-1)
    """
    confidence = base_confidence

    # Reduce confidence for imputed data
    if row.get('_is_imputed', False):
        confidence *= 0.7  # 30% penalty for imputed data

    # Scale by EMA completeness
    ema_completeness = row.get('_ema_completeness', 1.0)
    confidence *= (0.5 + 0.5 * ema_completeness)  # 50% minimum, scaling up

    return min(1.0, max(0.0, confidence))


# Convenience function for pipeline integration
def prepare_features_pipeline(
    df: pd.DataFrame,
    model_features: List[str],
    min_completeness: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame, DataQualityReport]:
    """
    Prepare features for model training/prediction pipeline.

    Splits data into complete and imputed subsets for potential
    differential handling.

    Args:
        df: Raw feature DataFrame
        model_features: Features required by the model
        min_completeness: Minimum completeness to include row

    Returns:
        Tuple of (complete_df, imputed_df, quality_report)
    """
    # Validate and impute
    df_prepared, report = validate_features_for_prediction(
        df, model_features, allow_imputation=True
    )

    # Filter by minimum completeness
    df_prepared = df_prepared[df_prepared['_ema_completeness'] >= min_completeness]

    # Split into complete and imputed
    complete_mask = df_prepared['_has_complete_ema']
    complete_df = df_prepared[complete_mask].copy()
    imputed_df = df_prepared[~complete_mask].copy()

    logger.info(f"Split: {len(complete_df)} complete, {len(imputed_df)} imputed rows")

    return complete_df, imputed_df, report
