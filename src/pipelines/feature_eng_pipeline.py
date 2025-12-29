"""
Feature engineering pipeline for creating ML-ready features.

This pipeline orchestrates feature creation:
1. Load preprocessed data from data/02-preprocessed/
2. Clean and validate data
3. Create features using multiple engineers
4. Merge all features
5. Save to data/03-features/
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config_loader import Config
from src.features.loaders import ParquetDataLoader, MultiFileLoader
from src.features.cleaners import BasicDataCleaner, MatchDataCleaner, PlayerStatsDataCleaner, LineupsDataCleaner
from src.features.engineers import (
    TeamFormFeatureEngineer,
    TeamStatsFeatureEngineer,
    MatchOutcomeFeatureEngineer,
    HeadToHeadFeatureEngineer,
    ExponentialMovingAverageFeatureEngineer,
    ELORatingFeatureEngineer,
    PoissonFeatureEngineer,
    GoalDifferenceFeatureEngineer,
    HomeAwayFormFeatureEngineer,
    RestDaysFeatureEngineer,
    LeaguePositionFeatureEngineer,
    StreakFeatureEngineer,
    # V4 Feature Engineers
    FormationFeatureEngineer,
    CoachFeatureEngineer,
    LineupStabilityFeatureEngineer,
    StarPlayerFeatureEngineer,
    TeamRatingFeatureEngineer,
    KeyPlayerAbsenceFeatureEngineer,
    DisciplineFeatureEngineer,
    GoalTimingFeatureEngineer,
    SeasonPhaseFeatureEngineer,
    DerbyFeatureEngineer,
)
from src.features.merger import DataMerger

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Pipeline for feature engineering.

    Transforms preprocessed Parquet data into ML-ready feature matrices.
    """

    def __init__(self, config: Config):
        """
        Initialize the feature engineering pipeline.

        Args:
            config: Configuration object loaded from YAML
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

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
        multi_loader = MultiFileLoader(loader)

        all_matches = []
        all_player_stats = []
        all_lineups = []
        all_events = []

        for season in self.config.seasons:
            season_dir = self.config.get_preprocessed_season_dir(season)
            self.logger.info(f"Loading season {season} from {season_dir}")

            matches_path = season_dir / "matches.parquet"
            player_stats_path = season_dir / "player_stats.parquet"
            lineups_path = season_dir / "lineups.parquet"
            events_path = season_dir / "events.parquet"

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
        """Create features using configured feature engineers."""
        feature_dfs = []

        self.logger.info("Creating team form features...")
        form_engineer = TeamFormFeatureEngineer(n_matches=self.config.features.form_window)
        form_features = form_engineer.create_features(cleaned_data)
        feature_dfs.append(form_features)

        if self.config.features.include_h2h:
            self.logger.info("Creating head-to-head features...")
            h2h_engineer = HeadToHeadFeatureEngineer(n_h2h=5)
            h2h_features = h2h_engineer.create_features(cleaned_data)
            feature_dfs.append(h2h_features)

        self.logger.info("Creating EMA features...")
        ema_engineer = ExponentialMovingAverageFeatureEngineer(span=self.config.features.ema_span)
        ema_features = ema_engineer.create_features(cleaned_data)
        feature_dfs.append(ema_features)

        if self.config.features.include_team_stats and 'player_stats' in cleaned_data:
            self.logger.info("Creating team stats EMA features...")
            try:
                stats_engineer = TeamStatsFeatureEngineer(span=self.config.features.ema_span)
                stats_features = stats_engineer.create_features(cleaned_data)
                if not stats_features.empty:
                    feature_dfs.append(stats_features)
            except Exception as e:
                self.logger.warning(f"Could not create team stats features: {e}")

        # New advanced features
        self.logger.info("Creating ELO rating features...")
        elo_engineer = ELORatingFeatureEngineer(k_factor=32.0, home_advantage=100.0)
        elo_features = elo_engineer.create_features(cleaned_data)
        feature_dfs.append(elo_features)

        self.logger.info("Creating Poisson-based features...")
        poisson_engineer = PoissonFeatureEngineer(lookback_matches=10)
        poisson_features = poisson_engineer.create_features(cleaned_data)
        feature_dfs.append(poisson_features)

        self.logger.info("Creating goal difference features...")
        gd_engineer = GoalDifferenceFeatureEngineer(lookback_matches=5)
        gd_features = gd_engineer.create_features(cleaned_data)
        feature_dfs.append(gd_features)

        # New features v3
        self.logger.info("Creating home/away form features...")
        home_away_engineer = HomeAwayFormFeatureEngineer(n_matches=self.config.features.form_window)
        home_away_features = home_away_engineer.create_features(cleaned_data)
        feature_dfs.append(home_away_features)

        self.logger.info("Creating rest days features...")
        rest_engineer = RestDaysFeatureEngineer()
        rest_features = rest_engineer.create_features(cleaned_data)
        feature_dfs.append(rest_features)

        self.logger.info("Creating league position features...")
        position_engineer = LeaguePositionFeatureEngineer()
        position_features = position_engineer.create_features(cleaned_data)
        feature_dfs.append(position_features)

        self.logger.info("Creating streak features...")
        streak_engineer = StreakFeatureEngineer()
        streak_features = streak_engineer.create_features(cleaned_data)
        feature_dfs.append(streak_features)

        # =====================================================================
        # V4 Features - Requires lineups, events, player_stats
        # =====================================================================

        self.logger.info("Creating formation features...")
        try:
            formation_engineer = FormationFeatureEngineer()
            formation_features = formation_engineer.create_features(cleaned_data)
            if not formation_features.empty and len(formation_features.columns) > 1:
                feature_dfs.append(formation_features)
        except Exception as e:
            self.logger.warning(f"Could not create formation features: {e}")

        self.logger.info("Creating coach features...")
        try:
            coach_engineer = CoachFeatureEngineer(lookback_matches=5)
            coach_features = coach_engineer.create_features(cleaned_data)
            if not coach_features.empty and len(coach_features.columns) > 1:
                feature_dfs.append(coach_features)
        except Exception as e:
            self.logger.warning(f"Could not create coach features: {e}")

        self.logger.info("Creating lineup stability features...")
        try:
            lineup_engineer = LineupStabilityFeatureEngineer(
                lookback_matches=self.config.features.lineup_lookback
            )
            lineup_features = lineup_engineer.create_features(cleaned_data)
            if not lineup_features.empty and len(lineup_features.columns) > 1:
                feature_dfs.append(lineup_features)
        except Exception as e:
            self.logger.warning(f"Could not create lineup stability features: {e}")

        self.logger.info("Creating star player features...")
        try:
            star_engineer = StarPlayerFeatureEngineer(
                top_n=self.config.features.star_top_n,
                min_matches=self.config.features.star_min_matches
            )
            star_features = star_engineer.create_features(cleaned_data)
            if not star_features.empty and len(star_features.columns) > 1:
                feature_dfs.append(star_features)
        except Exception as e:
            self.logger.warning(f"Could not create star player features: {e}")

        self.logger.info("Creating team rating features...")
        try:
            rating_engineer = TeamRatingFeatureEngineer(
                lookback_matches=self.config.features.rating_lookback
            )
            rating_features = rating_engineer.create_features(cleaned_data)
            if not rating_features.empty and len(rating_features.columns) > 1:
                feature_dfs.append(rating_features)
        except Exception as e:
            self.logger.warning(f"Could not create team rating features: {e}")

        self.logger.info("Creating key player absence features...")
        try:
            absence_engineer = KeyPlayerAbsenceFeatureEngineer(
                top_n=self.config.features.key_player_top_n,
                lookback_matches=self.config.features.rating_lookback
            )
            absence_features = absence_engineer.create_features(cleaned_data)
            if not absence_features.empty and len(absence_features.columns) > 1:
                feature_dfs.append(absence_features)
        except Exception as e:
            self.logger.warning(f"Could not create key player absence features: {e}")

        self.logger.info("Creating discipline features...")
        try:
            discipline_engineer = DisciplineFeatureEngineer(
                lookback_matches=self.config.features.discipline_lookback
            )
            discipline_features = discipline_engineer.create_features(cleaned_data)
            if not discipline_features.empty and len(discipline_features.columns) > 1:
                feature_dfs.append(discipline_features)
        except Exception as e:
            self.logger.warning(f"Could not create discipline features: {e}")

        self.logger.info("Creating goal timing features...")
        try:
            timing_engineer = GoalTimingFeatureEngineer(
                lookback_matches=self.config.features.goal_timing_lookback
            )
            timing_features = timing_engineer.create_features(cleaned_data)
            if not timing_features.empty and len(timing_features.columns) > 1:
                feature_dfs.append(timing_features)
        except Exception as e:
            self.logger.warning(f"Could not create goal timing features: {e}")

        self.logger.info("Creating season phase features...")
        try:
            phase_engineer = SeasonPhaseFeatureEngineer()
            phase_features = phase_engineer.create_features(cleaned_data)
            if not phase_features.empty and len(phase_features.columns) > 1:
                feature_dfs.append(phase_features)
        except Exception as e:
            self.logger.warning(f"Could not create season phase features: {e}")

        self.logger.info("Creating derby features...")
        try:
            derby_engineer = DerbyFeatureEngineer()
            derby_features = derby_engineer.create_features(cleaned_data)
            if not derby_features.empty and len(derby_features.columns) > 1:
                feature_dfs.append(derby_features)
        except Exception as e:
            self.logger.warning(f"Could not create derby features: {e}")

        self.logger.info("Creating target variables...")
        outcome_engineer = MatchOutcomeFeatureEngineer()
        outcome_features = outcome_engineer.create_features(cleaned_data)
        feature_dfs.append(outcome_features)

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
        """Save final features to CSV."""
        output_dir = self.config.get_features_dir()
        output_path = output_dir / output_filename

        final_data.to_csv(output_path, index=False)
        self.logger.info(f"Saved features to: {output_path}")

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
