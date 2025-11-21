"""Data writing utilities for preprocessing."""
import logging
from pathlib import Path

import pandas as pd

from src.preprocessing.exceptions import DataProcessingError
from src.preprocessing.interfaces import IDataWriter


class ParquetDataWriter(IDataWriter):
    """Writer for Parquet files."""

    def write(self, data: pd.DataFrame, path: Path) -> None:
        """Writes data to a Parquet file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(path, index=False)
            logging.info(f"Saved {len(data)} rows to {path}")
        except Exception as e:
            raise DataProcessingError(f"Error writing to {path}: {e}")
