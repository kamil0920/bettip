"""
Feature Interaction Discovery

Uses xgbfir to rank feature pair interactions by gain,
then engineers top interactions as multiplicative features.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)


def discover_interactions(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    is_regression: bool = False,
    top_n: int = 20,
    output_path: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """Discover top feature interactions using xgbfir.

    Trains an XGBoost model, then uses xgbfir to extract feature
    interaction pairs ranked by average gain.

    Args:
        X: Feature matrix.
        y: Target array.
        feature_names: Names of features matching X columns.
        is_regression: Whether this is a regression task.
        top_n: Number of top interaction pairs to return.
        output_path: Optional path to save xgbfir Excel output.

    Returns:
        List of (feature_a, feature_b, gain) tuples sorted by gain descending.
    """
    import xgbfir

    logger.info(f"Discovering feature interactions (top {top_n})")

    if is_regression:
        model = XGBRegressor(
            n_estimators=200, max_depth=6, random_state=42,
            verbosity=0, n_jobs=-1,
        )
    else:
        model = XGBClassifier(
            n_estimators=200, max_depth=6, random_state=42,
            verbosity=0, n_jobs=-1, eval_metric='logloss',
        )

    df_train = pd.DataFrame(X, columns=feature_names)
    model.fit(df_train, y)

    # xgbfir saves to file, so use a temp path if none provided
    if output_path is None:
        import tempfile
        output_path = tempfile.mktemp(suffix='.xlsx')

    xgbfir.saveXgbFI(model, feature_names=feature_names, OutputXlsxFile=output_path)

    # Read depth-2 interactions (sheet "Interaction Depth 1" = pairwise)
    try:
        interactions_df = pd.read_excel(output_path, sheet_name='Interaction Depth 1')
    except Exception:
        # Some versions use different sheet names
        interactions_df = pd.read_excel(output_path, sheet_name=1)

    if interactions_df.empty:
        logger.warning("No interactions found")
        return []

    # Extract top interactions
    top_interactions = []
    for _, row in interactions_df.head(top_n).iterrows():
        interaction_name = row.iloc[0]  # e.g. "feat_a|feat_b"
        gain = row.get('Average Gain', row.get('Gain', 0))

        parts = str(interaction_name).split('|')
        if len(parts) == 2:
            top_interactions.append((parts[0].strip(), parts[1].strip(), float(gain)))

    logger.info(f"Top 5 interactions: {[(a, b) for a, b, _ in top_interactions[:5]]}")
    return top_interactions


def engineer_interaction_features(
    df: pd.DataFrame,
    interactions: List[Tuple[str, str, float]],
    max_features: int = 20,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create multiplicative interaction features.

    Args:
        df: DataFrame with original features.
        interactions: List of (feat_a, feat_b, gain) from discover_interactions.
        max_features: Maximum number of interaction features to create.

    Returns:
        Tuple of (DataFrame with new columns, list of new column names).
    """
    new_cols = []
    df = df.copy()

    for feat_a, feat_b, gain in interactions[:max_features]:
        if feat_a not in df.columns or feat_b not in df.columns:
            continue

        col_name = f"ix_{feat_a}_x_{feat_b}"
        df[col_name] = df[feat_a] * df[feat_b]
        new_cols.append(col_name)

    logger.info(f"Engineered {len(new_cols)} interaction features")
    return df, new_cols
