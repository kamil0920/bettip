from typing import List

import pandas as pd

class DataMerger:
    """Class for merge features"""

    def merge_all_features(self, base_df: pd.DataFrame,
                          feature_dfs: List[pd.DataFrame],
                          merge_key: str = 'fixture_id') -> pd.DataFrame:
        """
        Merge into one DataFrame

        Args:
            base_df: base DataFrame
            feature_dfs: list of DataFrames with features
            merge_key: key

        Returns:
            Merged DataFrame
        """
        result = base_df.copy()

        for feature_df in feature_dfs:
            result = result.merge(feature_df, on=merge_key, how='left')

        print(f"✓ Merge all features: {len(result)} rows, {len(result.columns)} columns")
        return result
