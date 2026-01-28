"""Data I/O utilities for Parquet/CSV format handling."""

import logging
from pathlib import Path
from typing import Union

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
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        logger.info(f"Loading CSV (fallback): {csv_path}")
        return pd.read_csv(csv_path, low_memory=False)

    raise FileNotFoundError(f"No features file found: {parquet_path} or {csv_path}")


def save_features(df: pd.DataFrame, path: Union[str, Path], dual_format: bool = True) -> Path:
    """Save features DataFrame, writing Parquet as primary format.

    Args:
        df: DataFrame to save.
        path: Output path (extension will be adjusted).
        dual_format: If True, write both .parquet and .csv for transition period.

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
