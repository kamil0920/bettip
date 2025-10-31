from typing import Dict, List, Optional

import pandas as pd

from merger import DataMerger
from interfaces import IDataLoader, IDataCleaner, IFeatureEngineer
from data_loaders import MultiFileLoader
from data_cleaners import BasicDataCleaner


class SoccerDataPipeline:
    """
    Main pipeline
    """

    def __init__(self,
                 data_loader: IDataLoader,
                 cleaners: Dict[str, IDataCleaner],
                 feature_engineers: List[IFeatureEngineer],
                 merger: DataMerger):
        """
        Args:
            data_loader: object to load data
            cleaners: dict of cleaners
            feature_engineers: list of feature engineers
            merger: object to merge features
        """
        self.data_loader = data_loader
        self.cleaners = cleaners
        self.feature_engineers = feature_engineers
        self.merger = merger

    def process(self, filepaths: Dict[str, str],
                output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process data in pipeline

        Args:
            filepaths: paths to input files
            output_path: path to output file

        Returns:
            Processed data ready to train model
        """
        print("=" * 60)
        print("SOCCER DATA PREPROCESSING PIPELINE")
        print("=" * 60)

        print("\n[1/4] Load data...")
        multi_loader = MultiFileLoader(self.data_loader)
        raw_data = multi_loader.load_all(filepaths)

        print("\n[2/4] Clean data...")
        cleaned_data = {}
        for name, df in raw_data.items():
            if name in self.cleaners:
                cleaned_data[name] = self.cleaners[name].clean(df)
            else:
                cleaned_data[name] = BasicDataCleaner().clean(df)

        print("\n[3/4] Create features...")
        feature_dfs = []
        for engineer in self.feature_engineers:
            try:
                features = engineer.create_features(cleaned_data)
                feature_dfs.append(features)
            except Exception as e:
                print(f"✗ Error in {engineer.__class__.__name__}: {str(e)}")

        print("\n[4/4] Merge features...")
        base_df = cleaned_data['matches'][['fixture_id', 'date', 'home_team_id',
                                           'home_team_name', 'away_team_id',
                                           'away_team_name', 'round']]
        final_data = self.merger.merge_all_features(base_df, feature_dfs)

        initial_rows = len(final_data)
        final_data = final_data.dropna(subset=['home_wins_last_n', 'away_wins_last_n'])
        removed = initial_rows - len(final_data)

        print(f"\n✓ Removed {removed} rows with missing values")
        print(f"✓ Final data: {len(final_data)} rows, {len(final_data.columns)} columns")

        if output_path:
            final_data.to_csv(output_path, index=False)
            print(f"✓ Save to: {output_path}")

        print("\n" + "=" * 60)
        print("PIPELINE FINISHED!")
        print("=" * 60)

        return final_data
