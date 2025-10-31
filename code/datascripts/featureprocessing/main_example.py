from main import SoccerDataPipeline
from data_loaders import ParquetDataLoader
from merger import DataMerger
from feature_engineering import (TeamFormFeatureEngineer,
                                 TeamStatsFeatureEngineer,
                                 HeadToHeadFeatureEngineer,
                                 ExponentialMovingAverageFeatureEngineer,
                                 MatchOutcomeFeatureEngineer)
from data_cleaners import MatchDataCleaner, PlayerStatsDataCleaner, BasicDataCleaner
import pandas as pd
import os
from pathlib import Path


def main():
    """Run pipeline for all years"""

    base_path = Path("../processed_data/premier_league")

    years = [2020, 2021, 2022, 2023, 2024, 2025]

    all_data = []

    cleaners = {
        'matches': MatchDataCleaner(),
        'player_stats': PlayerStatsDataCleaner(),
        'events': BasicDataCleaner(),
        'lineups': BasicDataCleaner(),
        'teams': BasicDataCleaner()
    }

    feature_engineers = [
        TeamFormFeatureEngineer(n_matches=5),
        TeamStatsFeatureEngineer(),
        HeadToHeadFeatureEngineer(n_h2h=3),
        ExponentialMovingAverageFeatureEngineer(span=10),
        MatchOutcomeFeatureEngineer()
    ]

    pipeline = SoccerDataPipeline(
        data_loader=ParquetDataLoader(),
        cleaners=cleaners,
        feature_engineers=feature_engineers,
        merger=DataMerger()
    )

    for year in years:
        print(f"\n{'=' * 50}")
        print(f"Processing year: {year}")
        print(f"{'=' * 50}")

        year_path = base_path / str(year)
        filepaths = {
            'matches': str(year_path / 'matches.parquet'),
            'player_stats': str(year_path / 'player_stats.parquet'),
            'events': str(year_path / 'events.parquet'),
            'lineups': str(year_path / 'lineups.parquet'),
            'teams': str(year_path / 'teams.parquet')
        }

        missing_files = [k for k, v in filepaths.items() if not os.path.exists(v)]
        if missing_files:
            print(f"Warning: Missing files for {year}: {missing_files}")
            continue

        try:
            year_data = pipeline.process(
                filepaths=filepaths,
                output_path=None
            )

            year_data['season'] = year

            all_data.append(year_data)
            print(f"Successfully processed {len(year_data)} rows for {year}")

        except Exception as e:
            print(f"Error processing {year}: {str(e)}")
            continue

    if all_data:
        print(f"\n{'=' * 50}")
        print("Combining all years...")
        print(f"{'=' * 50}")

        final_data = pd.concat(all_data, ignore_index=True)

        output_file = '../advanced_features/football_ml_data_all_years.csv'

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        final_data.to_csv(output_file, index=False)
        print(f"\nSaved combined data to: {output_file}")

        print(f"\n{'=' * 50}")
        print("FINAL DATA SUMMARY")
        print(f"{'=' * 50}")
        print(f"\nTotal rows: {len(final_data)}")
        print(f"\nRows per season:")
        print(final_data['season'].value_counts().sort_index())

        print("\nFirst 5 rows:")
        print(final_data.head())

        print("\nData info:")
        print(final_data.info())

        print("\nDescriptive statistics:")
        print(final_data.describe().round(3))

        return final_data
    else:
        print("\nNo data was processed!")
        return None


if __name__ == "__main__":
    final_data = main()
