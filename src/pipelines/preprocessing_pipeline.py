"""
Preprocessing pipeline for raw data to structured data transformation.

This pipeline orchestrates the ETL process:
1. Load raw JSON data from data/01-raw/
2. Extract, validate, and transform data
3. Save structured Parquet files to data/02-preprocessed/
"""
import logging
from typing import Dict

import pandas as pd

from src.config_loader import Config
from src.preprocessing.factory import DataProcessorFactory

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Pipeline for preprocessing raw football data.

    Transforms raw JSON API data into structured Parquet files
    suitable for feature engineering.
    """

    def __init__(self, config: Config):
        """
        Initialize the preprocessing pipeline.

        Args:
            config: Configuration object loaded from YAML
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the preprocessing pipeline.

        Returns:
            Dictionary with processed DataFrames:
            - 'matches': Match/fixture data
            - 'events': Match events
            - 'players': Player statistics
            - 'lineups': Team lineups
            - 'teams': Unique teams

        Raises:
            FileNotFoundError: If raw data directories don't exist
            DataProcessingError: If processing fails
        """
        self.logger.info("=" * 60)
        self.logger.info("PREPROCESSING PIPELINE")
        self.logger.info("=" * 60)

        self.logger.info("[1/3] Validating configuration...")
        if not DataProcessorFactory.validate_config(self.config):
            raise FileNotFoundError(
                f"Configuration validation failed. "
                f"Check that raw data exists in {self.config.data.raw_dir}"
            )

        self.logger.info("[2/3] Creating data processor...")
        processor = DataProcessorFactory.create_data_processor(self.config)

        self.logger.info("[3/3] Processing seasons...")
        self.logger.info(f"Seasons to process: {self.config.seasons}")
        self.logger.info(f"League: {self.config.league}")
        self.logger.info(f"Output directory: {self.config.data.preprocessed_dir}")

        result = processor.process_all_seasons()

        self._log_summary(result)

        return result

    def _log_summary(self, result: Dict[str, pd.DataFrame]) -> None:
        """Log pipeline execution summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PREPROCESSING PIPELINE COMPLETED")
        self.logger.info("=" * 60)

        for name, df in result.items():
            if not df.empty:
                self.logger.info(f"{name}: {len(df)} rows, {len(df.columns)} columns")
            else:
                self.logger.info(f"{name}: empty")

        self.logger.info(f"\nData saved to: {self.config.data.preprocessed_dir}")
        self.logger.info("=" * 60)
