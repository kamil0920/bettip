"""Data I/O utilities for Parquet/CSV format handling."""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_features(path: Union[str, Path]) -> pd.DataFrame:
    """Load features file, preferring Parquet over CSV.

    Args:
        path: Path to features file (.csv or .parquet).
              If a .csv path is given, tries .parquet first.

    Returns:
        DataFrame with loaded features.

    Raises:
        FileNotFoundError: If neither Parquet nor CSV file exists.
    """
    path = Path(path)
    parquet_path = path.with_suffix('.parquet')
    csv_path = path.with_suffix('.csv')

    if parquet_path.exists():
        logger.info(f"Loading Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        logger.info(f"Loading CSV (fallback): {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        raise FileNotFoundError(f"No features file found: {parquet_path} or {csv_path}")

    # Clean object columns with bracketed scientific notation (e.g. '[5.07E-1]')
    # from legacy parquet files where _clean_for_parquet stringified numeric values.
    # Without this, SHAP validation fails for shots/corners.
    text_cols = {'date', 'home_team', 'away_team', 'league', 'season', 'referee'}
    n_cleaned = 0
    for col in df.select_dtypes(include='object').columns:
        if col in text_cols:
            continue
        converted = pd.to_numeric(
            df[col].astype(str).str.strip('[]() '), errors='coerce'
        )
        if converted.notna().sum() >= df[col].notna().sum() * 0.5:
            df[col] = converted
            n_cleaned += 1
    if n_cleaned > 0:
        logger.info(f"Cleaned {n_cleaned} bracketed-string columns to numeric")

    return df


def save_features(df: pd.DataFrame, path: Union[str, Path], dual_format: bool = False) -> Path:
    """Save features DataFrame, writing Parquet as primary format.

    Args:
        df: DataFrame to save.
        path: Output path (extension will be adjusted).
        dual_format: If True, write both .parquet and .csv.

    Returns:
        Path to the primary (Parquet) output file.
    """
    path = Path(path)
    parquet_path = path.with_suffix('.parquet')
    csv_path = path.with_suffix('.csv')

    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved Parquet: {parquet_path}")

    if dual_format:
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV (dual-format): {csv_path}")

    return parquet_path
