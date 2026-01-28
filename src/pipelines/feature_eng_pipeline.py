"""
Feature engineering pipeline for creating ML-ready features.

Uses the Registry Pattern for feature engineers:
- Centralized configuration of all engineers
- Easy to add/remove features without modifying pipeline code
- Configuration-driven customization

This pipeline orchestrates feature creation:
1. Load preprocessed data from data/02-preprocessed/
2. Clean and validate data
3. Create features using registered engineers
4. Merge all features
5. Save to data/03-features/
"""
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config_loader import Config
from src.features.loaders import ParquetDataLoader, MultiFileLoader
from src.features.cleaners import MatchDataCleaner, PlayerStatsDataCleaner, LineupsDataCleaner
from src.features.merger import DataMerger
from src.features.registry import get_registry, get_default_configs

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Pipeline for feature engineering.

    Uses Registry pattern - feature engineers are registered centrally
    and instantiated based on configuration.
    """

    TARGET_COLUMNS = [
        'home_win', 'draw', 'away_win', 'match_result',
        'total_goals', 'goal_difference', 'gd_form_diff'
    ]

    def __init__(self, config: Config):
        """
        Initialize the feature engineering pipeline.

        Args:
            config: Configuration object loaded from YAML
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry = get_registry()

    def run(self, output_filename: str = "features.csv") -> pd.DataFrame:
        """
        Execute the feature engineering pipeline.

        Args:
            output_filename: Name of output CSV file

        Returns:
            DataFrame with all features merged

        Raises:
            FileNotFoundError: If preprocessed data doesn't exist
            Exception: If feature engineering fails
        """
        self.logger.info("=" * 60)
        self.logger.info("FEATURE ENGINEERING PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Available engineers: {self.registry.list_engineers()}")

        self.logger.info("[1/5] Loading preprocessed data...")
        raw_data = self._load_data()

        self.logger.info("[2/5] Cleaning data...")
        cleaned_data = self._clean_data(raw_data)

        self.logger.info("[3/5] Creating features...")
        feature_dfs = self._create_features(cleaned_data)

        self.logger.info("[4/5] Merging features...")
        final_data = self._merge_features(cleaned_data, feature_dfs)

        self.logger.info("[5/5] Saving results...")
        output_path = self._save_results(final_data, output_filename)

        self._log_summary(final_data, output_path)

        return final_data

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all preprocessed data for configured seasons."""
        loader = ParquetDataLoader()

        all_matches = []
        all_player_stats = []
        all_lineups = []
        all_events = []
        all_match_stats = []

        for season in self.config.seasons:
            season_dir = self.config.get_preprocessed_season_dir(season)
            raw_season_dir = self.config.get_raw_season_dir(season)
            self.logger.info(f"Loading season {season} from {season_dir}")

            matches_path = season_dir / "matches.parquet"
            player_stats_path = season_dir / "player_stats.parquet"
            lineups_path = season_dir / "lineups.parquet"
            events_path = season_dir / "events.parquet"
            # match_stats is in raw directory, not preprocessed
            match_stats_path = raw_season_dir / "match_stats.parquet"

            if not matches_path.exists():
                self.logger.warning(f"Matches file not found: {matches_path}")
                continue

            matches_df = loader.load(str(matches_path))
            all_matches.append(matches_df)

            if player_stats_path.exists():
                player_stats_df = loader.load(str(player_stats_path))
                all_player_stats.append(player_stats_df)

            if lineups_path.exists():
                lineups_df = loader.load(str(lineups_path))
                all_lineups.append(lineups_df)

            if events_path.exists():
                events_df = loader.load(str(events_path))
                all_events.append(events_df)

            if match_stats_path.exists():
                match_stats_df = loader.load(str(match_stats_path))
                all_match_stats.append(match_stats_df)

        if not all_matches:
            raise FileNotFoundError(
                f"No matches data found in {self.config.data.preprocessed_dir}"
            )

        combined_matches = pd.concat(all_matches, ignore_index=True)
        self.logger.info(f"Loaded {len(combined_matches)} matches total")

        result = {'matches': combined_matches}

        if all_player_stats:
            combined_player_stats = pd.concat(all_player_stats, ignore_index=True)
            result['player_stats'] = combined_player_stats
            self.logger.info(f"Loaded {len(combined_player_stats)} player stats total")

        if all_lineups:
            combined_lineups = pd.concat(all_lineups, ignore_index=True)
            result['lineups'] = combined_lineups
            self.logger.info(f"Loaded {len(combined_lineups)} lineup entries total")

        if all_events:
            combined_events = pd.concat(all_events, ignore_index=True)
            result['events'] = combined_events
            self.logger.info(f"Loaded {len(combined_events)} events total")

        if all_match_stats:
            combined_match_stats = pd.concat(all_match_stats, ignore_index=True)
            result['match_stats'] = combined_match_stats
            self.logger.info(f"Loaded {len(combined_match_stats)} match stats total")

        return result

    def _clean_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean all data using appropriate cleaners."""
        cleaned_data = {}

        match_cleaner = MatchDataCleaner()
        cleaned_data['matches'] = match_cleaner.clean(raw_data['matches'])

        team_name_to_id = self._build_team_mapping(cleaned_data['matches'])

        if 'player_stats' in raw_data:
            player_cleaner = PlayerStatsDataCleaner()
            cleaned_data['player_stats'] = player_cleaner.clean(raw_data['player_stats'])
            cleaned_data['player_stats'] = self._map_team_ids(
                cleaned_data['player_stats'], team_name_to_id
            )

        if 'lineups' in raw_data:
            lineups_cleaner = LineupsDataCleaner()
            cleaned_data['lineups'] = lineups_cleaner.clean(raw_data['lineups'])
            cleaned_data['lineups'] = self._map_team_ids(
                cleaned_data['lineups'], team_name_to_id
            )

        if 'events' in raw_data:
            cleaned_data['events'] = raw_data['events']

        if 'match_stats' in raw_data:
            cleaned_data['match_stats'] = raw_data['match_stats']
            # Merge match_stats into matches for feature engineers (cards, fouls, corners)
            stats_cols = ['fixture_id', 'home_yellow_cards', 'away_yellow_cards',
                          'home_red_cards', 'away_red_cards', 'home_fouls', 'away_fouls',
                          'home_corner_kicks', 'away_corner_kicks', 'home_total_shots',
                          'away_total_shots', 'home_shots_on_goal', 'away_shots_on_goal']
            available_cols = [c for c in stats_cols if c in raw_data['match_stats'].columns]
            if available_cols:
                match_stats_subset = raw_data['match_stats'][available_cols].drop_duplicates(subset=['fixture_id'])
                cleaned_data['matches'] = cleaned_data['matches'].merge(
                    match_stats_subset, on='fixture_id', how='left'
                )
                self.logger.info(f"Merged {len(available_cols)-1} match stats columns into matches")

        return cleaned_data

    def _build_team_mapping(self, matches: pd.DataFrame) -> dict:
        """Build mapping from team names to team IDs."""
        team_map = {}
        if 'home_team_id' in matches.columns and 'home_team_name' in matches.columns:
            for _, row in matches[['home_team_id', 'home_team_name']].drop_duplicates().iterrows():
                team_map[row['home_team_name']] = row['home_team_id']
            for _, row in matches[['away_team_id', 'away_team_name']].drop_duplicates().iterrows():
                team_map[row['away_team_name']] = row['away_team_id']
        return team_map

    def _map_team_ids(self, df: pd.DataFrame, team_map: dict) -> pd.DataFrame:
        """Map team_name strings to team_id integers if needed."""
        if df.empty or not team_map:
            return df

        if 'team_id' in df.columns and df['team_id'].dtype == 'object':
            df = df.copy()
            df['team_id'] = df['team_id'].map(team_map)
            original_len = len(df)
            df = df.dropna(subset=['team_id'])
            if len(df) < original_len:
                self.logger.debug(f"Dropped {original_len - len(df)} rows with unmapped team names")
            df['team_id'] = df['team_id'].astype(int)

        if 'team_name' in df.columns and 'team_id' not in df.columns:
            df = df.copy()
            df['team_id'] = df['team_name'].map(team_map)
            df = df.dropna(subset=['team_id'])
            df['team_id'] = df['team_id'].astype(int)

        return df

    def _create_features(self, cleaned_data: Dict[str, pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Create features using registry.

        Uses the Registry pattern - all feature engineers are registered
        centrally and configured via FeatureEngineerConfig.
        """
        configs = get_default_configs(self.config)

        feature_dfs = self.registry.create_all_features(
            cleaned_data,
            configs,
            on_error='warn'
        )

        return feature_dfs

    def _merge_features(self, cleaned_data: Dict[str, pd.DataFrame], feature_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge all features into single DataFrame."""
        merger = DataMerger()

        base_df = cleaned_data['matches'][[
            'fixture_id', 'date', 'home_team_id', 'home_team_name',
            'away_team_id', 'away_team_name', 'round'
        ]]

        final_data = merger.merge_all_features(base_df, feature_dfs)

        initial_rows = len(final_data)
        final_data = final_data.dropna(subset=['home_wins_last_n', 'away_wins_last_n'])
        removed = initial_rows - len(final_data)

        if removed > 0:
            self.logger.info(f"Removed {removed} rows with missing form features")

        return final_data

    def _save_results(self, final_data: pd.DataFrame, output_filename: str) -> Path:
        """
        Save final features to Parquet (primary) and CSV (backward compatible).

        Creates three file pairs:
        1. {output_filename} - Full file with features + targets
        2. {output_filename}_features_only - Features without target columns
        3. {output_filename}_targets - Target columns only

        This separation helps prevent accidental data leakage in experiments.
        """
        from src.utils.data_io import save_features

        output_dir = self.config.get_features_dir()
        output_path = output_dir / output_filename

        save_features(final_data, output_path, dual_format=True)
        self.logger.info(f"Saved full features to: {output_path}")

        targets_present = [col for col in self.TARGET_COLUMNS if col in final_data.columns]

        if targets_present:
            feature_cols = [col for col in final_data.columns if col not in self.TARGET_COLUMNS]
            base_name = output_filename.replace('.csv', '').replace('.parquet', '')
            features_only_path = output_dir / f"{base_name}_features_only"
            save_features(final_data[feature_cols], features_only_path, dual_format=True)
            self.logger.info(f"Saved features-only to: {features_only_path}")

            id_cols = ['fixture_id', 'date']
            target_cols = id_cols + targets_present
            targets_path = output_dir / f"{base_name}_targets"
            save_features(final_data[target_cols], targets_path, dual_format=True)
            self.logger.info(f"Saved targets to: {targets_path}")

            self.logger.warning(
                "NOTE: Use *_features_only files for training to prevent data leakage. "
                f"Target columns ({targets_present}) are saved separately in *_targets files"
            )

        return output_path

    def _log_summary(self, final_data: pd.DataFrame, output_path: Path) -> None:
        """Log pipeline execution summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FEATURE ENGINEERING PIPELINE COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Final dataset: {len(final_data)} rows, {len(final_data.columns)} columns")
        self.logger.info(f"Output saved to: {output_path}")

        feature_cols = [
            col for col in final_data.columns
            if col not in ['fixture_id', 'date', 'home_team_id', 'home_team_name',
                           'away_team_id', 'away_team_name', 'round']
        ]
        self.logger.info(f"Total features: {len(feature_cols)}")
        self.logger.info("=" * 60)
