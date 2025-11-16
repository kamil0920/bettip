"""Feature merging utilities."""
from typing import List

import pandas as pd


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

        Args:
            base_df: base DataFrame
            feature_dfs: list of DataFrames with features
            merge_key: key to merge on

        Returns:
            Merged DataFrame
        """
        result = base_df.copy()

        for feature_df in feature_dfs:
            result = result.merge(feature_df, on=merge_key, how='left')

        print(f"Merged all features: {len(result)} rows, {len(result.columns)} columns")
        return result
