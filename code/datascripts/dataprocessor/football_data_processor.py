#!/usr/bin/env python3
"""
FootballDataProcessor integrating existing robust modules
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import logging

# Import your existing modules
from normalization.helpers import sanitize_for_write, first_notna, maybe_load_json, to_int_safe
from normalization.extractors import extract_fixture_core, extract_events_from_row, extract_player_stats_from_row
from normalization.normalizers import normalize_player_stats_df, normalize_stats_per_90_by_position
from normalization.process_player_stats import build_player_form_features, add_player_ema

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballDataProcessor:
    """
    Data processor using existing robust modules.
    Processes raw football data into ML-ready features with advanced player analytics.
    """

    def __init__(self, data_dir: str = None, processed_dir: str = None):
        if data_dir is None:
            # Jeśli uruchamiasz z code/datascripts/dataprocessor/
            self.data_dir = Path("../apicalls/football_data")
        else:
            self.data_dir = Path(data_dir)

        if processed_dir is None:
            self.processed_dir = Path("../../../processed_data")
        else:
            self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_all_seasons(self, league: str = "premier_league",
                            seasons: List[int] = None,
                            include_player_features: bool = True,
                            include_events: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Process all seasons with your existing robust pipeline.

        Returns:
            Dict containing processed DataFrames: matches, events, players, teams
        """
        if seasons is None:
            seasons = [2020, 2021, 2022, 2023, 2024]  # Skip incomplete 2025

        logger.info(f"🚀 Processing {len(seasons)} seasons: {seasons}")

        # Collect data from all seasons
        all_matches = []
        all_events = []
        all_players = []
        all_teams = []

        for season in seasons:
            logger.info(f"📅 Processing season {season}...")

            season_data = self.process_season_with_existing_pipeline(
                league, season, include_player_features, include_events
            )

            if season_data:
                all_matches.append(season_data['matches'])
                all_events.append(season_data['events'])
                all_players.append(season_data['players'])
                all_teams.append(season_data['teams'])

        # Combine all seasons
        combined_data = {}

        if all_matches:
            combined_data['matches'] = pd.concat(all_matches, ignore_index=True)
            combined_data['matches'] = self._enhance_match_features(combined_data['matches'])

        if all_events:
            combined_data['events'] = pd.concat(all_events, ignore_index=True)

        if all_players:
            combined_data['players'] = pd.concat(all_players, ignore_index=True)
            if include_player_features:
                combined_data['players'] = self._enhance_player_features(combined_data['players'])

        if all_teams:
            combined_data['teams'] = pd.concat(all_teams, ignore_index=True).drop_duplicates()

        # Create ML dataset
        if 'matches' in combined_data:
            combined_data['ml_dataset'] = self._create_ml_dataset(
                combined_data['matches'],
                combined_data.get('players'),
                combined_data.get('events')
            )

        logger.info(f"✅ Processing completed!")
        self._log_data_summary(combined_data)

        return combined_data

    def process_season_with_existing_pipeline(self, league: str, season: int,
                                              include_players: bool = True,
                                              include_events: bool = True) -> Optional[Dict]:
        """Use your existing robust processing pipeline for a single season."""

        season_dir = self.data_dir / league / str(season)

        if not season_dir.exists():
            logger.warning(f"Season directory not found: {season_dir}")
            return None

        # Step 1: Load raw fixture data using your structure
        fixtures_data = self._load_season_fixtures(season_dir)
        if not fixtures_data:
            return None

        # Step 2: Extract using your extractors
        matches_rows = []
        events_rows = []
        players_rows = []
        teams_map = {}

        logger.info(f"Processing {len(fixtures_data)} fixtures for season {season}")

        for idx, row in fixtures_data.iterrows():
            try:
                # Use your extract_fixture_core
                match_data = extract_fixture_core(row)
                if match_data.get('fixture_id'):
                    matches_rows.append(match_data)

                    # Collect teams
                    if match_data.get("home_team_id"):
                        teams_map[match_data["home_team_id"]] = {
                            "team_id": match_data["home_team_id"],
                            "team_name": match_data.get("home_team_name")
                        }
                    if match_data.get("away_team_id"):
                        teams_map[match_data["away_team_id"]] = {
                            "team_id": match_data["away_team_id"],
                            "team_name": match_data.get("away_team_name")
                        }

                # Extract events using your extractor
                if include_events:
                    events = extract_events_from_row(row)
                    if events:
                        events_rows.extend(events)

                # Extract players using your extractor
                if include_players:
                    # DODAJ TĘ SEKCJĘ:
                    lineups_dir = season_dir / "lineups"
                    events_dir = season_dir / "events"

                    if lineups_dir.exists():
                        logger.info(f"Found lineups directory with {len(list(lineups_dir.glob('*.json')))} files")
                        # Process lineup files instead of using extractors
                        lineup_files = list(lineups_dir.glob("fixture_*_lineups.json"))
                        for lineup_file in lineup_files:
                            try:
                                with open(lineup_file, 'r') as f:
                                    lineup_data = json.load(f)
                                    if 'data' in lineup_data and 'lineups' in lineup_data['data']:
                                        # Extract players from lineups
                                        lineups = lineup_data['data']['lineups']
                                        fixture_id = lineup_data['data']['fixture_info']['id']

                                        for team_lineup in lineups:
                                            team_id = team_lineup['team']['id']

                                            # Process starting XI
                                            for player in team_lineup.get('startXI', []):
                                                player_info = player['player']
                                                players_rows.append({
                                                    'fixture_id': fixture_id,
                                                    'team_id': team_id,
                                                    'player_id': player_info['id'],
                                                    'player_name': player_info['name'],
                                                    'position': player_info['pos'],
                                                    'number': player_info['number'],
                                                    'starting': True
                                                })

                                            # Process substitutes
                                            for player in team_lineup.get('substitutes', []):
                                                player_info = player['player']
                                                players_rows.append({
                                                    'fixture_id': fixture_id,
                                                    'team_id': team_id,
                                                    'player_id': player_info['id'],
                                                    'player_name': player_info['name'],
                                                    'position': player_info['pos'],
                                                    'number': player_info['number'],
                                                    'starting': False
                                                })
                            except Exception as e:
                                logger.warning(f"Error processing lineup file {lineup_file}: {e}")
                    else:
                        logger.warning(f"No lineups directory found in {season_dir}")

                    if include_events and events_dir.exists():
                        logger.info(f"Found events directory with {len(list(events_dir.glob('*.json')))} files")
                        # Similar processing for events
                        event_files = list(events_dir.glob("fixture_*_events.json"))
                        for event_file in event_files:
                            try:
                                with open(event_file, 'r') as f:
                                    event_data = json.load(f)
                                    if 'data' in event_data and 'events' in event_data['data']:
                                        events = event_data['data']['events']
                                        fixture_id = event_data['data']['fixture_info']['id']

                                        for event in events:
                                            events_rows.append({
                                                'fixture_id': fixture_id,
                                                'type': event.get('type'),
                                                'detail': event.get('detail'),
                                                'time': event.get('time', {}).get('elapsed'),
                                                'team_id': event.get('team', {}).get('id'),
                                                'player_id': event.get('player', {}).get('id') if event.get(
                                                    'player') else None,
                                                'assist_id': event.get('assist', {}).get('id') if event.get(
                                                    'assist') else None
                                            })
                            except Exception as e:
                                logger.warning(f"Error processing event file {event_file}: {e}")
                    elif include_events:
                        logger.warning(f"No events directory found in {season_dir}")

            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue

        # Step 3: Create DataFrames
        matches_df = pd.DataFrame(matches_rows)
        events_df = pd.DataFrame(events_rows) if events_rows else pd.DataFrame()
        players_df = pd.DataFrame(players_rows) if players_rows else pd.DataFrame()
        teams_df = pd.DataFrame(list(teams_map.values())) if teams_map else pd.DataFrame()

        # Step 4: Apply your normalizers
        if not players_df.empty:
            players_df = normalize_player_stats_df(players_df)

            # Add position-aware performance metrics
            if 'games_position' in players_df.columns:
                players_df = normalize_stats_per_90_by_position(players_df, drop_old_features=False)

            # Add player form features with EMA
            if not matches_df.empty:
                players_df = build_player_form_features(
                    players_df, matches_df, keep_lead_cols=True, drop_old_features=False
                )

        return {
            'matches': matches_df,
            'events': events_df,
            'players': players_df,
            'teams': teams_df
        }

    def _load_season_fixtures(self, season_dir: Path) -> Optional[pd.DataFrame]:
        """Load fixtures data from your existing structure."""

        # Try different possible file names from your structure
        possible_files = [
            'fixtures.json',
            'fixtures_detailed_all.parquet'
        ]

        for pattern in possible_files:
            if '*' in pattern:
                files = list(season_dir.glob(pattern))
                if files:
                    # Take most recent
                    fixtures_file = sorted(files)[-1]
                    break
            else:
                fixtures_file = season_dir / pattern
                if fixtures_file.exists():
                    break
        else:
            logger.warning(f"No fixtures file found in {season_dir}")
            return None

        try:
            if fixtures_file.suffix == '.json':
                with open(fixtures_file, 'r') as f:
                    data = json.load(f)
                    fixtures = data.get('data', [])
                    return pd.DataFrame(fixtures)
            elif fixtures_file.suffix == '.parquet':
                return pd.read_parquet(fixtures_file)
            else:
                logger.warning(f"Unsupported file format: {fixtures_file}")
                return None

        except Exception as e:
            logger.error(f"Error loading {fixtures_file}: {e}")
            return None

    def _enhance_match_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced match-level features."""
        df = matches_df.copy()

        logger.info(f"Available columns: {list(df.columns)}")

        # Add target variables
        home_goals_col = 'ft_home' if 'ft_home' in df.columns else 'home_goals'
        away_goals_col = 'ft_away' if 'ft_away' in df.columns else 'away_goals'

        df['total_goals'] = (pd.to_numeric(df.get(home_goals_col, 0), errors='coerce').fillna(0) +
                             pd.to_numeric(df.get(away_goals_col, 0), errors='coerce').fillna(0))

        df['goal_difference'] = (pd.to_numeric(df.get(home_goals_col, 0), errors='coerce').fillna(0) -
                                 pd.to_numeric(df.get(away_goals_col, 0), errors='coerce').fillna(0))

        home_scores = pd.to_numeric(df.get(home_goals_col, 0), errors='coerce').fillna(0) > 0
        away_scores = pd.to_numeric(df.get(away_goals_col, 0), errors='coerce').fillna(0) > 0

        # Both teams score
        df['both_teams_score'] = (home_scores & away_scores).astype(int)

        # Match result
        df['home_win'] = (df['goal_difference'] > 0).astype(int)
        df['draw'] = (df['goal_difference'] == 0).astype(int)
        df['away_win'] = (df['goal_difference'] < 0).astype(int)

        # Over/Under markets
        df['over_1_5'] = (df['total_goals'] > 1.5).astype(int)
        df['over_2_5'] = (df['total_goals'] > 2.5).astype(int)
        df['over_3_5'] = (df['total_goals'] > 3.5).astype(int)

        # Convert date to datetime if it's not already
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        logger.info(f"Enhanced matches: {len(df)} with {len(df.columns)} features")
        return df

    def _enhance_player_features(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Apply additional player feature enhancements using your modules."""
        if players_df.empty:
            return players_df

        df = players_df.copy()

        # Apply your advanced EMA features if not already done
        if not any(col.endswith('_ema5') for col in df.columns):
            df = add_player_ema(df, span=5)

        # Add team aggregations
        if 'fixture_id' in df.columns and 'team_id' in df.columns:
            team_aggs = df.groupby(['fixture_id', 'team_id']).agg({
                'rating_ema5': 'mean',
                'minutes_share_ema5': 'mean',
                'performance_per_90': 'mean'
            }).add_prefix('team_avg_')

            df = df.merge(team_aggs, left_on=['fixture_id', 'team_id'],
                          right_index=True, how='left')

        logger.info(f"Enhanced players: {len(df)} with {len(df.columns)} features")
        return df

    def _create_ml_dataset(self, matches_df: pd.DataFrame,
                           players_df: Optional[pd.DataFrame] = None,
                           events_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create final ML dataset combining all features."""

        ml_df = matches_df.copy()

        # Add player aggregations per team per match
        if players_df is not None and not players_df.empty:
            player_features = self._create_team_player_features(players_df)
            ml_df = ml_df.merge(player_features, on='fixture_id', how='left')

        # Add event features
        if events_df is not None and not events_df.empty:
            event_features = self._create_event_features(events_df)
            ml_df = ml_df.merge(event_features, on='fixture_id', how='left')

        # Add team form features
        ml_df = self._add_team_form_features(ml_df)

        # Clean and finalize
        ml_df = self._clean_ml_features(ml_df)

        logger.info(f"Created ML dataset: {len(ml_df)} matches with {len(ml_df.columns)} features")
        return ml_df

    def _create_team_player_features(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player features by team and fixture."""

        # Group by fixture and team
        team_features = players_df.groupby(['fixture_id', 'team_id']).agg({
            'rating_ema5': ['mean', 'std', 'min', 'max'],
            'minutes_share_ema5': ['mean', 'sum'],
            'performance_per_90': ['mean', 'std'],
        }).round(3)

        # Flatten column names
        team_features.columns = ['_'.join(col).strip() for col in team_features.columns]
        team_features = team_features.reset_index()

        # Pivot to get home/away team features
        home_features = team_features.copy()
        away_features = team_features.copy()

        home_features.columns = ['fixture_id', 'home_team_id'] + [f'home_{col}' for col in home_features.columns[2:]]
        away_features.columns = ['fixture_id', 'away_team_id'] + [f'away_{col}' for col in away_features.columns[2:]]

        # Merge home and away
        fixture_features = home_features.merge(away_features, on='fixture_id', how='outer')

        return fixture_features[
            ['fixture_id'] + [col for col in fixture_features.columns if col.startswith(('home_', 'away_'))]]

    def _create_event_features(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from match events."""

        event_features = events_df.groupby('fixture_id').agg({
            'type': 'count',  # total events
        }).rename(columns={'type': 'total_events'})

        # Count specific event types
        for event_type in ['Goal', 'Card', 'subst']:
            mask = events_df['type'] == event_type
            type_counts = events_df[mask].groupby('fixture_id').size()
            event_features[f'{event_type.lower()}_events'] = type_counts

        return event_features.fillna(0).reset_index()

    def _add_team_form_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Add team form features (last N games)."""

        if 'date' not in df.columns:
            logger.warning("No date column found, skipping form features")
            return df

        df = df.sort_values('date').copy()

        # This is a simplified version - your existing pipeline might have more sophisticated form calculation
        for team_col in ['home_team_id', 'away_team_id']:
            prefix = team_col.split('_')[0]  # 'home' or 'away'

            # Calculate rolling team performance
            team_results = []

            for idx, row in df.iterrows():
                team_id = row[team_col]
                match_date = row['date']

                # Get previous matches for this team
                prev_matches = df[(df['date'] < match_date) &
                                  ((df['home_team_id'] == team_id) | (df['away_team_id'] == team_id))].tail(window)

                if len(prev_matches) > 0:
                    # Calculate form metrics
                    points = 0
                    goals_for = 0
                    goals_against = 0

                    for _, prev_match in prev_matches.iterrows():
                        if prev_match['home_team_id'] == team_id:
                            gf = prev_match.get('home_goals', prev_match.get('ft_home', 0))
                            ga = prev_match.get('away_goals', prev_match.get('ft_away', 0))
                        else:
                            gf = prev_match.get('away_goals', prev_match.get('ft_away', 0))
                            ga = prev_match.get('home_goals', prev_match.get('ft_home', 0))

                        goals_for += gf if pd.notna(gf) else 0
                        goals_against += ga if pd.notna(ga) else 0

                        if gf > ga:
                            points += 3
                        elif gf == ga:
                            points += 1

                    form_points = points
                    form_goals_for = goals_for
                    form_goals_against = goals_against
                else:
                    form_points = form_goals_for = form_goals_against = 0

                team_results.append({
                    f'{prefix}_form_points': form_points,
                    f'{prefix}_form_goals_for': form_goals_for,
                    f'{prefix}_form_goals_against': form_goals_against,
                    f'{prefix}_form_matches': len(prev_matches)
                })

            # Add to dataframe
            form_df = pd.DataFrame(team_results)
            for col in form_df.columns:
                df[col] = form_df[col].values

        return df

    def _clean_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize ML features."""

        # Fill NaN values appropriately
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Remove rows with missing key identifiers
        df = df.dropna(subset=['fixture_id', 'home_team_id', 'away_team_id'])

        # Ensure proper data types
        id_cols = ['fixture_id', 'home_team_id', 'away_team_id', 'venue_id', 'league_id']
        for col in id_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        return df

    def save_processed_data(self, data: Dict[str, pd.DataFrame], timestamp: str = None):
        """Save all processed datasets."""

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_files = {}

        for dataset_name, df in data.items():
            if df is not None and not df.empty:
                filename = f"{dataset_name}_{timestamp}.parquet"
                filepath = self.processed_dir / filename

                # Apply your sanitization
                df_clean = sanitize_for_write(df)
                df_clean.to_parquet(filepath, index=False)

                saved_files[dataset_name] = filepath
                logger.info(f"💾 Saved {dataset_name}: {filepath} ({len(df)} rows, {len(df.columns)} cols)")

        return saved_files

    def _log_data_summary(self, data: Dict[str, pd.DataFrame]):
        """Log summary of processed data."""
        logger.info("📊 DATA PROCESSING SUMMARY")
        logger.info("=" * 50)

        for name, df in data.items():
            if df is not None and not df.empty:
                logger.info(f"{name.upper()}: {len(df)} rows, {len(df.columns)} columns")
                if name == 'ml_dataset':
                    # Show key statistics for ML dataset
                    if 'total_goals' in df.columns:
                        logger.info(f"  Avg goals per match: {df['total_goals'].mean():.2f}")
                    if 'home_win' in df.columns:
                        logger.info(f"  Home win rate: {df['home_win'].mean():.1%}")
                    if 'over_2_5' in df.columns:
                        logger.info(f"  Over 2.5 goals rate: {df['over_2_5'].mean():.1%}")


if __name__ == "__main__":
    import os

    print(f"Current working directory: {os.getcwd()}")
    print(f"Data directory exists: {Path('../apicalls/football_data').exists()}")
    print(f"Premier League folder exists: {Path('../apicalls/football_data/premier_league').exists()}")

    processor = FootballDataProcessor()

    # Process all data
    processed_data = processor.process_all_seasons(
        league="premier_league",
        seasons=[2020, 2021, 2022, 2023, 2024, 2025],
        include_player_features=True,
        include_events=True
    )

    # Save processed data
    if processed_data:
        saved_files = processor.save_processed_data(processed_data)
        print(f"✅ Processing completed! Saved {len(saved_files)} datasets")
    else:
        print("❌ No data processed")
