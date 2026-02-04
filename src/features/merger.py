"""Feature merging utilities."""
import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class DataMerger:
    """Class for merging features."""

    def merge_all_features(
            self,
            base_df: pd.DataFrame,
            feature_dfs: List[pd.DataFrame],
            merge_key: str = 'fixture_id'
    ) -> pd.DataFrame:
        """
        Merge all feature DataFrames into one.

        Detects and resolves duplicate column names before merging to prevent
        pandas from creating _x/_y suffixed columns (which cause data leakage
        when Boruta selects these leaked duplicates).

        Args:
            base_df: base DataFrame
            feature_dfs: list of DataFrames with features
            merge_key: key to merge on

        Returns:
            Merged DataFrame
        """
        result = base_df.copy()

        for feature_df in feature_dfs:
            # Detect overlapping columns (excluding merge key)
            overlap = set(result.columns) & set(feature_df.columns) - {merge_key}
            if overlap:
                logger.warning(
                    f"Dropping {len(overlap)} duplicate columns from later engineer: "
                    f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
                )
                feature_df = feature_df.drop(columns=overlap)

            result = result.merge(feature_df, on=merge_key, how='left')

        # Safety check: flag any _x/_y columns that slipped through
        xy_cols = [c for c in result.columns if c.endswith('_x') or c.endswith('_y')]
        if xy_cols:
            logger.error(
                f"LEAKAGE WARNING: {len(xy_cols)} columns with _x/_y suffixes detected: "
                f"{xy_cols[:10]}. These indicate unresolved merge conflicts."
            )

        logger.info(f"Merged all features: {len(result)} rows, {len(result.columns)} columns")
        return result
