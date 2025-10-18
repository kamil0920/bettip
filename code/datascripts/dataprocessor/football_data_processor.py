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
        Process all seasons using your existing robust normalize_season pipeline.
        """
        if seasons is None:
            seasons = [2020, 2021, 2022, 2023, 2024]  # Skip incomplete 2025

        logger.info(f"🚀 Processing {len(seasons)} seasons: {seasons}")

        # Collect data from all seasons
        all_matches = []
        all_events = []
        all_players = []
        all_teams = []

        # Track successful seasons
        successful_seasons = []

        for season in seasons:
            logger.info(f"📅 Processing season {season}...")

            season_data = self.process_season_with_normalize_season(
                league, season, include_player_features, include_events
            )

            if season_data:
                all_matches.append(season_data['matches'])
                all_events.append(season_data['events'])
                all_players.append(season_data['players'])
                all_teams.append(season_data['teams'])
                successful_seasons.append(season)
            else:
                logger.warning(f"❌ Failed to process season {season}")

        logger.info(f"✅ Successfully processed {len(successful_seasons)} seasons: {successful_seasons}")

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

    def process_season_with_normalize_season(self, league: str, season: int,
                                             include_players: bool = True,
                                             include_events: bool = True) -> Optional[Dict]:
        """Process season using YOUR JSON structure - creates player stats from events!"""

        season_dir = self.data_dir / league / str(season)
        season_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Processing JSON files for season {season}")

            # Load fixtures.json
            fixtures_file = season_dir / "fixtures.json"
            if not fixtures_file.exists():
                logger.error(f"fixtures.json not found for season {season}")
                return None

            with open(fixtures_file, 'r') as f:
                fixtures_data = json.load(f)

            # Extract matches using your extractors
            matches_rows = []
            teams_map = {}

            fixtures = fixtures_data.get('data', [])
            logger.info(f"Processing {len(fixtures)} fixtures for season {season}")

            for fixture in fixtures:
                try:
                    status = fixture.get('fixture', {}).get('status', {}).get('short', '')
                    if status not in ['FT', 'AET', 'PEN']:
                        logger.debug(f"Skipping fixture {fixture.get('fixture', {}).get('id')} with status: {status}")
                        continue

                    fixture_series = pd.Series(fixture)
                    match_data = extract_fixture_core(fixture_series)
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
                except Exception as e:
                    logger.warning(f"Error processing fixture: {e}")
                    continue

            # Create matches DataFrame
            matches_df = pd.DataFrame(matches_rows)
            teams_df = pd.DataFrame(list(teams_map.values())) if teams_map else pd.DataFrame()

            # STEP 1: Load all events and create player stats from them
            events_rows = []
            player_stats_from_events = {}  # player_id + fixture_id -> stats

            if include_events:
                events_dir = season_dir / "events"
                if events_dir.exists():
                    event_files = list(events_dir.glob("fixture_*_events.json"))
                    logger.info(f"Processing {len(event_files)} event files to create player stats")

                    for event_file in event_files:
                        try:
                            with open(event_file, 'r') as f:
                                event_data = json.load(f)

                            if 'data' in event_data and 'events' in event_data['data']:
                                events = event_data['data']['events']
                                fixture_id = event_data['data']['fixture_info']['id']

                                for event in events:
                                    # Add to events list
                                    events_rows.append({
                                        'fixture_id': fixture_id,
                                        'type': event.get('type'),
                                        'detail': event.get('detail'),
                                        'time': event.get('time', {}).get('elapsed'),
                                        'team_id': event.get('team', {}).get('id'),
                                        'player_id': event.get('player', {}).get('id') if event.get('player') else None,
                                        'assist_id': event.get('assist', {}).get('id') if event.get('assist') else None,
                                        'raw': json.dumps(event_data)
                                    })

                                    # Extract player stats from events
                                    player_id = event.get('player', {}).get('id') if event.get('player') else None
                                    team_id = event.get('team', {}).get('id')
                                    assist_id = event.get('assist', {}).get('id') if event.get('assist') else None

                                    if player_id:
                                        key = f"{fixture_id}_{player_id}"
                                        if key not in player_stats_from_events:
                                            player_stats_from_events[key] = {
                                                'fixture_id': fixture_id,
                                                'player_id': player_id,
                                                'team_id': team_id,
                                                'player_name': event.get('player', {}).get('name'),
                                                'goals': 0,
                                                'assists': 0,
                                                'yellow_cards': 0,
                                                'red_cards': 0,
                                                'minutes': 90,  # Default, will be updated from lineups
                                                'position': None,  # Will be updated from lineups
                                                'starting': None,  # Will be updated from lineups
                                            }

                                        # Count stats based on event type
                                        stats = player_stats_from_events[key]
                                        if event.get('type') == 'Goal' and event.get('detail') not in ['Own Goal',
                                                                                                       'Penalty cancelled']:
                                            stats['goals'] += 1
                                        elif event.get('type') == 'Card':
                                            if event.get('detail') == 'Yellow Card':
                                                stats['yellow_cards'] += 1
                                            elif event.get('detail') in ['Red Card', 'Second Yellow card']:
                                                stats['red_cards'] += 1

                                    # Count assists
                                    if assist_id:
                                        key_assist = f"{fixture_id}_{assist_id}"
                                        if key_assist not in player_stats_from_events:
                                            player_stats_from_events[key_assist] = {
                                                'fixture_id': fixture_id,
                                                'player_id': assist_id,
                                                'team_id': team_id,
                                                'player_name': event.get('assist', {}).get('name'),
                                                'goals': 0,
                                                'assists': 0,
                                                'yellow_cards': 0,
                                                'red_cards': 0,
                                                'minutes': 90,
                                                'position': None,
                                                'starting': None,
                                            }
                                        if event.get('type') == 'Goal' and event.get('detail') not in ['Own Goal']:
                                            player_stats_from_events[key_assist]['assists'] += 1

                        except Exception as e:
                            logger.warning(f"Error processing event file {event_file}: {e}")

            # STEP 2: Load lineups and merge with event stats
            players_rows = []
            if include_players:
                # First, try to load from API player statistics
                players_dir = season_dir / "players"
                has_api_stats = players_dir.exists() and len(list(players_dir.glob("fixture_*_players.json"))) > 0

                if has_api_stats:
                    logger.info(f"Loading player statistics from API files...")

                    # Get all fixture IDs from matches
                    fixture_ids = matches_df['fixture_id'].unique() if not matches_df.empty else []

                    for fixture_id in fixture_ids:
                        api_stats = self._load_player_statistics_from_api(season_dir, fixture_id)

                        if api_stats:
                            # Merge with event-based stats if available
                            for player_stat in api_stats:
                                player_id = player_stat['player_id']
                                key = f"{fixture_id}_{player_id}"

                                # Get event-based goals/assists if available
                                event_stats = player_stats_from_events.get(key, {})

                                # Prefer API data but fill gaps with event data
                                if event_stats:
                                    # Use event data for goals/assists if API has 0
                                    if player_stat['goals'] == 0 and event_stats.get('goals', 0) > 0:
                                        player_stat['goals'] = event_stats['goals']
                                    if player_stat['assists'] == 0 and event_stats.get('assists', 0) > 0:
                                        player_stat['assists'] = event_stats['assists']

                                    # Always use event data for cards (more reliable)
                                    player_stat['yellow_cards'] = event_stats.get('yellow_cards',
                                                                                  player_stat['yellow_cards'])
                                    player_stat['red_cards'] = event_stats.get('red_cards', player_stat['red_cards'])

                                players_rows.append(player_stat)

                    logger.info(f"Loaded {len(players_rows)} player records from API statistics")

                else:
                    # Fallback: Use lineups + event stats (old method)
                    logger.info(f"No API player statistics found, using lineups + events fallback...")
                    lineups_dir = season_dir / "lineups"

                    if lineups_dir.exists():
                        lineup_files = list(lineups_dir.glob("fixture_*_lineups.json"))
                        logger.info(f"Found {len(lineup_files)} lineup files")

                        for lineup_file in lineup_files:
                            try:
                                with open(lineup_file, 'r') as f:
                                    lineup_data = json.load(f)

                                if 'data' in lineup_data and 'lineups' in lineup_data['data']:
                                    lineups = lineup_data['data']['lineups']
                                    fixture_id = lineup_data['data']['fixture_info']['id']

                                    for team_lineup in lineups:
                                        team_id = team_lineup['team']['id']

                                        # Process starting XI
                                        for player in team_lineup.get('startXI', []):
                                            player_info = player['player']
                                            player_id = player_info['id']
                                            key = f"{fixture_id}_{player_id}"

                                            # Get stats from events or create basic record
                                            stats = player_stats_from_events.get(key, {
                                                'fixture_id': fixture_id,
                                                'player_id': player_id,
                                                'team_id': team_id,
                                                'player_name': player_info['name'],
                                                'goals': 0,
                                                'assists': 0,
                                                'yellow_cards': 0,
                                                'red_cards': 0,
                                                'minutes': 90,
                                            })

                                            # Update with lineup info
                                            stats.update({
                                                'position': player_info['pos'],
                                                'number': player_info['number'],
                                                'starting': True,
                                                'minutes': 90,
                                                'raw': json.dumps(lineup_data)
                                            })

                                            players_rows.append(stats)

                                        # Process substitutes
                                        for player in team_lineup.get('substitutes', []):
                                            player_info = player['player']
                                            player_id = player_info['id']
                                            key = f"{fixture_id}_{player_id}"

                                            stats = player_stats_from_events.get(key, {
                                                'fixture_id': fixture_id,
                                                'player_id': player_id,
                                                'team_id': team_id,
                                                'player_name': player_info['name'],
                                                'goals': 0,
                                                'assists': 0,
                                                'yellow_cards': 0,
                                                'red_cards': 0,
                                                'minutes': 0,
                                            })

                                            stats.update({
                                                'position': player_info['pos'],
                                                'number': player_info['number'],
                                                'starting': False,
                                                'minutes': 0,
                                                'raw': json.dumps(lineup_data)
                                            })

                                            players_rows.append(stats)

                            except Exception as e:
                                logger.warning(f"Error processing lineup file {lineup_file}: {e}")

            # Create DataFrames
            events_df = pd.DataFrame(events_rows) if events_rows else pd.DataFrame()
            players_df = pd.DataFrame(players_rows) if players_rows else pd.DataFrame()

            # Now you have players_df with: goals, assists, yellow_cards, red_cards, minutes, position, starting!
            # Apply normalization
            if not players_df.empty:
                try:
                    logger.info(f"Applying normalization to {len(players_df)} player records")

                    # Dodaj fixture_dt (wymagane przez EMA)
                    if 'fixture_dt' not in players_df.columns:
                        # Merge dates from matches
                        date_mapping = matches_df.set_index('fixture_id')['date'].to_dict()
                        players_df['fixture_dt'] = players_df['fixture_id'].map(date_mapping)
                        players_df['fixture_dt'] = pd.to_datetime(players_df['fixture_dt'])

                    # Dodaj games_rating jeśli nie ma rating
                    # if 'games_rating' not in players_df.columns:
                    #     if 'rating' in players_df.columns:
                    #         players_df['games_rating'] = players_df['rating']
                    #     else:
                    #         # Fallback: create basic rating
                    #         players_df['games_rating'] = (
                    #                 6.0 +
                    #                 players_df.get('goals', 0) * 0.3 +
                    #                 players_df.get('assists', 0) * 0.2 -
                    #                 players_df.get('yellow_cards', 0) * 0.1 -
                    #                 players_df.get('red_cards', 0) * 0.5
                    #         ).clip(1.0, 10.0)

                    if 'games_position' not in players_df.columns and 'position' in players_df.columns:
                        players_df['games_position'] = players_df['position']

                    if 'minutes_share' not in players_df.columns:
                        players_df['minutes_share'] = (players_df['minutes'] / 90.0).clip(0, 1)

                    if not matches_df.empty:
                        try:
                            players_df = build_player_form_features(
                                players_df, matches_df, keep_lead_cols=True, drop_old_features=False
                            )
                        except Exception as e:
                            logger.warning(f"Error in build_player_form_features: {e}")

                    if 'fixture_dt' in players_df.columns and 'player_id' in players_df.columns:
                        try:
                            original_cols_to_keep = [
                                'goals', 'assists', 'yellow_cards', 'red_cards',
                                'shots_total', 'shots_on', 'passes_total', 'passes_key',
                                'tackles_total', 'duels_won', 'dribbles_success',
                                'position', 'starting', 'captain', 'substitute'
                            ]

                            original_data = {}
                            for col in original_cols_to_keep:
                                if col in players_df.columns:
                                    original_data[col] = players_df[col].copy()

                            players_df = add_player_ema(players_df, span=5)

                            for col, data in original_data.items():
                                if col not in players_df.columns:
                                    players_df[col] = data

                            logger.info(f"✅ Applied EMA features and restored {len(original_data)} original columns")
                        except Exception as e:
                            logger.warning(f"Skipping EMA features: {e}")

                    logger.info(f"✅ Normalization completed. Final columns: {len(players_df.columns)}")

                except Exception as e:
                    logger.warning(f"Error in player normalization: {e}")
                    logger.info("Continuing with raw API data")

            logger.info(f"✅ Successfully processed season {season}")
            logger.info(f"   Matches: {len(matches_df)} rows")
            logger.info(f"   Events: {len(events_df)} rows")
            logger.info(f"   Players: {len(players_df)} rows")
            logger.info(f"   Teams: {len(teams_df)} rows")

            return {
                'matches': matches_df,
                'events': events_df,
                'players': players_df,
                'teams': teams_df
            }

        except Exception as e:
            logger.error(f"Error processing season {season}: {e}")
            return None

    def _load_player_statistics_from_api(self, season_dir: Path, fixture_id: int) -> Optional[Dict]:
        """Load player statistics from API-sourced JSON files (with rating, shots, passes, etc.)"""
        players_dir = season_dir / "players"
        if not players_dir.exists():
            return None

        players_file = players_dir / f"fixture_{fixture_id}_players.json"
        if not players_file.exists():
            return None

        try:
            with open(players_file, 'r') as f:
                data = json.load(f)

            if 'data' not in data or 'players' not in data['data']:
                return None

            # Extract player statistics from API structure
            player_stats = []

            for team_data in data['data']['players']:
                team_id = team_data['team']['id']
                team_name = team_data['team']['name']

                for player_entry in team_data['players']:
                    player = player_entry['player']
                    stats = player_entry['statistics'][0] if player_entry['statistics'] else {}

                    # Extract comprehensive statistics
                    games = stats.get('games', {})
                    goals_data = stats.get('goals', {})
                    shots = stats.get('shots', {})
                    passes = stats.get('passes', {})
                    tackles = stats.get('tackles', {})
                    duels = stats.get('duels', {})
                    dribbles = stats.get('dribbles', {})
                    fouls = stats.get('fouls', {})
                    cards = stats.get('cards', {})
                    penalty = stats.get('penalty', {})

                    player_stat = {
                        'fixture_id': fixture_id,
                        'player_id': player['id'],
                        'player_name': player['name'],
                        'team_id': team_id,
                        'team_name': team_name,

                        # Game info
                        'minutes': games.get('minutes') or 0,
                        'position': games.get('position'),
                        'number': games.get('number'),
                        'rating': float(games.get('rating')) if games.get('rating') else None,
                        'captain': games.get('captain', False),
                        'substitute': games.get('substitute', False),
                        'starting': not games.get('substitute', False),

                        # Goals and assists
                        'goals': goals_data.get('total') or 0,
                        'assists': goals_data.get('assists') or 0,
                        'goals_conceded': goals_data.get('conceded') or 0,
                        'saves': goals_data.get('saves') or 0,

                        # Shots
                        'shots_total': shots.get('total') or 0,
                        'shots_on': shots.get('on') or 0,

                        # Passes
                        'passes_total': passes.get('total') or 0,
                        'passes_key': passes.get('key') or 0,
                        'passes_accuracy': passes.get('accuracy') or 0,

                        # Defensive
                        'tackles_total': tackles.get('total') or 0,
                        'tackles_blocks': tackles.get('blocks') or 0,
                        'tackles_interceptions': tackles.get('interceptions') or 0,

                        # Duels
                        'duels_total': duels.get('total') or 0,
                        'duels_won': duels.get('won') or 0,

                        # Dribbles
                        'dribbles_attempts': dribbles.get('attempts') or 0,
                        'dribbles_success': dribbles.get('success') or 0,
                        'dribbles_past': dribbles.get('past') or 0,

                        # Fouls
                        'fouls_drawn': fouls.get('drawn') or 0,
                        'fouls_committed': fouls.get('committed') or 0,

                        # Cards
                        'yellow_cards': cards.get('yellow', 0),
                        'red_cards': cards.get('red', 0),

                        # Penalties
                        'penalty_won': penalty.get('won') or 0,
                        'penalty_committed': penalty.get('commited') or 0,
                        'penalty_scored': penalty.get('scored', 0),
                        'penalty_missed': penalty.get('missed', 0),
                        'penalty_saved': penalty.get('saved', 0),

                        # Offsides
                        'offsides': stats.get('offsides') or 0
                    }

                    player_stats.append(player_stat)

            return player_stats

        except Exception as e:
            logger.warning(f"Error loading player statistics for fixture {fixture_id}: {e}")
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
            try:
                df = add_player_ema(df, span=5)
            except Exception as e:
                logger.warning(f"Error adding EMA features: {e}")

        # Add team aggregations - SAFE VERSION
        if 'fixture_id' in df.columns and 'team_id' in df.columns:
            # Check which columns exist before aggregating
            agg_cols = {}

            if 'rating_ema5' in df.columns:
                agg_cols['rating_ema5'] = 'mean'
            if 'minutes_share_ema5' in df.columns:
                agg_cols['minutes_share_ema5'] = 'mean'
            if 'performance_per_90' in df.columns:
                agg_cols['performance_per_90'] = 'mean'

            if agg_cols:
                try:
                    team_aggs = df.groupby(['fixture_id', 'team_id']).agg(agg_cols).add_prefix('team_avg_')
                    df = df.merge(team_aggs, left_on=['fixture_id', 'team_id'],
                                  right_index=True, how='left')
                except Exception as e:
                    logger.warning(f"Error creating team aggregations: {e}")

        logger.info(f"Enhanced players: {len(df)} with {len(df.columns)} features")
        return df

    def _create_ml_dataset(self, matches_df: pd.DataFrame,
                           players_df: Optional[pd.DataFrame] = None,
                           events_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create final ML dataset combining all features."""

        ml_df = matches_df.copy()

        # Add player aggregations per team per match - PASS matches_df!
        if players_df is not None and not players_df.empty:
            player_features = self._create_team_player_features(players_df, matches_df)
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

    def _create_team_player_features(self, players_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player features by team and fixture - CORRECTLY IDENTIFYING HOME/AWAY."""

        # Check which columns exist before aggregating
        agg_dict = {}

        if 'rating_ema5' in players_df.columns:
            agg_dict['rating_ema5'] = ['mean', 'std', 'min', 'max']
        if 'minutes_share_ema5' in players_df.columns:
            agg_dict['minutes_share_ema5'] = ['mean', 'sum']
        if 'performance_per_90' in players_df.columns:
            agg_dict['performance_per_90'] = ['mean', 'std']

        if not agg_dict:
            logger.warning("No advanced player columns found for team aggregation")
            return pd.DataFrame(columns=['fixture_id'])

        try:
            # Group by fixture and team
            team_features = players_df.groupby(['fixture_id', 'team_id']).agg(agg_dict).round(3)

            # Flatten column names
            team_features.columns = ['_'.join(col).strip() for col in team_features.columns]
            team_features = team_features.reset_index()

            # Store the feature column names (excluding fixture_id and team_id)
            feature_cols = [col for col in team_features.columns if col not in ['fixture_id', 'team_id']]

            # Get home/away team info from matches_df
            match_info = matches_df[['fixture_id', 'home_team_id', 'away_team_id']].copy()

            # Split into home and away features using actual team IDs
            home_features = team_features.merge(
                match_info[['fixture_id', 'home_team_id']],
                left_on=['fixture_id', 'team_id'],
                right_on=['fixture_id', 'home_team_id'],
                how='inner'
            )

            away_features = team_features.merge(
                match_info[['fixture_id', 'away_team_id']],
                left_on=['fixture_id', 'team_id'],
                right_on=['fixture_id', 'away_team_id'],
                how='inner'
            )

            # Select only the columns we need and rename with home/away prefixes
            home_cols = ['fixture_id'] + [f'home_{col}' for col in feature_cols]
            away_cols = ['fixture_id'] + [f'away_{col}' for col in feature_cols]

            # Use the stored feature_cols instead of referencing team_features.columns
            home_features = home_features[['fixture_id'] + feature_cols]
            away_features = away_features[['fixture_id'] + feature_cols]

            home_features.columns = home_cols
            away_features.columns = away_cols

            # Merge home and away features
            result = home_features.merge(away_features, on='fixture_id', how='outer')

            return result.fillna(0)

        except Exception as e:
            logger.warning(f"Error creating team player features: {e}")
            return pd.DataFrame(columns=['fixture_id'])

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
        """Add team form features (last N games) - FIXED VERSION."""

        if 'date' not in df.columns:
            logger.warning("No date column found, skipping form features")
            return df

        df = df.sort_values('date').copy()

        # Pre-calculate all team matches to avoid repeated filtering
        team_matches = {}
        for team_col in ['home_team_id', 'away_team_id']:
            for team_id in df[team_col].unique():
                if pd.notna(team_id):
                    team_mask = (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
                    team_matches[team_id] = df[team_mask].copy()

        # Calculate form features
        for team_col in ['home_team_id', 'away_team_id']:
            prefix = team_col.split('_')[0]  # 'home' or 'away'

            form_points = []
            form_goals_for = []
            form_goals_against = []
            form_matches_count = []

            # Use itertuples instead of iterrows for better performance
            for row in df.itertuples():
                team_id = getattr(row, team_col)
                match_date = row.date

                if team_id in team_matches:
                    prev_matches = team_matches[team_id][team_matches[team_id]['date'] < match_date].tail(window)

                    if len(prev_matches) > 0:
                        # Calculate goals and points for each previous match
                        match_points = 0
                        total_goals_for = 0
                        total_goals_against = 0

                        for _, prev_match in prev_matches.iterrows():
                            # Determine if team was home or away
                            if prev_match['home_team_id'] == team_id:
                                # Team was home
                                gf = prev_match.get('ft_home', prev_match.get('home_goals', 0))
                                ga = prev_match.get('ft_away', prev_match.get('away_goals', 0))
                            else:
                                # Team was away
                                gf = prev_match.get('ft_away', prev_match.get('away_goals', 0))
                                ga = prev_match.get('ft_home', prev_match.get('home_goals', 0))

                            # Handle NaN values
                            gf = gf if pd.notna(gf) else 0
                            ga = ga if pd.notna(ga) else 0

                            total_goals_for += gf
                            total_goals_against += ga

                            # Calculate points for this match
                            if gf > ga:
                                match_points += 3
                            elif gf == ga:
                                match_points += 1
                            # Loss = 0 points (no need to add)

                        form_points.append(match_points)
                        form_goals_for.append(total_goals_for)
                        form_goals_against.append(total_goals_against)
                        form_matches_count.append(len(prev_matches))
                    else:
                        form_points.append(0)
                        form_goals_for.append(0)
                        form_goals_against.append(0)
                        form_matches_count.append(0)
                else:
                    form_points.append(0)
                    form_goals_for.append(0)
                    form_goals_against.append(0)
                    form_matches_count.append(0)

            # Add to dataframe
            df[f'{prefix}_form_points'] = form_points
            df[f'{prefix}_form_goals_for'] = form_goals_for
            df[f'{prefix}_form_goals_against'] = form_goals_against
            df[f'{prefix}_form_matches'] = form_matches_count

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

                # DODAJ: Clean mixed-type columns BEFORE sanitization
                df_clean = df.copy()

                # Fix passes_accuracy if it exists (remove % and convert to float)
                if 'passes_accuracy' in df_clean.columns:
                    def clean_accuracy(val):
                        if pd.isna(val) or val is None:
                            return 0.0
                        if isinstance(val, str):
                            return float(val.replace('%', '').strip())
                        return float(val)

                    df_clean['passes_accuracy'] = df_clean['passes_accuracy'].apply(clean_accuracy)

                # Convert all object columns that should be numeric
                for col in df_clean.columns:
                    if df_clean[col].dtype == 'object':
                        # Try to convert to numeric
                        try:
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
                        except:
                            pass  # Keep as string if conversion fails

                # Apply your sanitization
                df_clean = sanitize_for_write(df_clean)
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
