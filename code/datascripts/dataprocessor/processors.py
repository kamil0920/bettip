import logging
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import pandas as pd

from config import ProcessingConfig
from interfaces import (
    IFixtureExtractor,
    IEventExtractor,
    IPlayerStatsExtractor,
    ILineupExtractor,
    IDataWriter
)
from loaders import (
    FixturesLoader,
    EventsLoader,
    LineupsLoader,
    PlayerStatsLoader
)

logger = logging.getLogger(__name__)


class SeasonDataProcessor:
    """
    Season data processor using specialized interfaces.
    Orchestrates the entire ETL process for football season data.

    Responsibilities:
    - Load raw data from JSON files
    - Extract and validate data using specialized extractors
    - Transform data into ML-ready format
    - Save processed data to Parquet files
    """

    def __init__(
            self,
            config: ProcessingConfig,
            fixtures_loader: FixturesLoader,
            events_loader: EventsLoader,
            lineups_loader: LineupsLoader,
            player_stats_loader: PlayerStatsLoader,
            fixture_extractor: IFixtureExtractor,
            event_extractor: IEventExtractor,
            player_extractor: IPlayerStatsExtractor,
            lineup_extractor: ILineupExtractor,
            writer: IDataWriter
    ):
        """
        Initialize the processor with all dependencies.
        Uses dependency injection pattern for loose coupling.

        Args:
            config: Processing configuration
            fixtures_loader: Loader for fixtures data
            events_loader: Loader for events data
            lineups_loader: Loader for lineups data
            player_stats_loader: Loader for player statistics
            fixture_extractor: Extractor for fixture data
            event_extractor: Extractor for event data
            player_extractor: Extractor for player statistics
            lineup_extractor: Extractor for lineup data
            writer: Writer for saving processed data
        """
        self.config = config
        self.fixtures_loader = fixtures_loader
        self.events_loader = events_loader
        self.lineups_loader = lineups_loader
        self.player_stats_loader = player_stats_loader
        self.fixture_extractor = fixture_extractor
        self.event_extractor = event_extractor
        self.player_extractor = player_extractor
        self.lineup_extractor = lineup_extractor
        self.writer = writer
        self.logger = logging.getLogger(self.__class__.__name__)

        # Statistics tracking
        self._stats = {
            'total_matches': 0,
            'total_events': 0,
            'total_players': 0,
            'total_lineups': 0,
            'failed_matches': 0,
            'failed_events': 0,
            'failed_players': 0,
            'failed_lineups': 0
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<SeasonDataProcessor(seasons={self.config.seasons}, "
            f"league={self.config.league})>"
        )

    def process_all_seasons(self) -> Dict[str, pd.DataFrame]:
        """
        Process all configured seasons and return combined results.

        This is the main entry point for processing multiple seasons.
        It processes each season individually and then combines the results.

        Returns:
            Dictionary with combined DataFrames:
            - 'matches': Combined matches from all seasons
            - 'events': Combined events from all seasons
            - 'players': Combined player stats from all seasons
            - 'lineups': Combined lineups from all seasons
            - 'teams': Combined unique teams from all seasons

        Raises:
            DataProcessingError: If processing fails based on error_handling config
        """
        self.logger.info(
            f"🚀 Starting processing for {len(self.config.seasons)} season(s): "
            f"{self.config.seasons}"
        )

        # Collections for combined data
        all_matches = []
        all_events = []
        all_players = []
        all_lineups = []
        all_teams = []

        # Process each season
        for season in self.config.seasons:
            try:
                season_data = self.process_season(season)

                if 'matches' in season_data and not season_data['matches'].empty:
                    all_matches.append(season_data['matches'])

                if 'events' in season_data and not season_data['events'].empty:
                    all_events.append(season_data['events'])

                if 'players' in season_data and not season_data['players'].empty:
                    all_players.append(season_data['players'])

                if 'lineups' in season_data and not season_data['lineups'].empty:
                    all_lineups.append(season_data['lineups'])

                if 'teams' in season_data and not season_data['teams'].empty:
                    all_teams.append(season_data['teams'])

            except Exception as e:
                self.logger.error(f"❌ Error processing season {season}: {e}")
                if self.config.error_handling == "raise":
                    raise

        combined = self._combine_seasons(
            all_matches, all_events, all_players, all_lineups, all_teams
        )

        self._log_final_stats()

        return combined

    def process_season(self, season: int) -> Dict[str, pd.DataFrame]:
        """
        Process a single season.

        This method orchestrates the entire ETL process for one season:
        1. Load fixtures
        2. Extract and validate data
        3. Save results

        Args:
            season: Season year to process

        Returns:
            Dictionary with DataFrames for this season
        """
        self.logger.info(f"🏁 Processing season {season}")

        season_dir = self.config.get_season_dir(season)

        # 1. Load fixtures
        fixtures = self.fixtures_loader.load_fixtures(season_dir)
        if not fixtures:
            self.logger.warning(f"⚠️  No fixtures found for season {season}")
            return {}

        # 2. Process each fixture
        matches_rows = []
        events_rows = []
        players_rows = []
        lineups_rows = []
        teams_map = {}

        for fixture in tqdm(fixtures, desc=f"Season {season}"):
            fixture_id = fixture.get('fixture', {}).get('id')

            if not fixture_id:
                continue

            # Extract match
            match = self.fixture_extractor.extract(fixture)
            if match:
                matches_rows.append(match)
                self._collect_teams(match, teams_map)
                self._stats['total_matches'] += 1
            else:
                self._stats['failed_matches'] += 1

            # Extract events if enabled
            if self.config.include_events:
                events_data = self.events_loader.load_events(season_dir, fixture_id)
                if events_data:
                    events = self.event_extractor.extract(events_data, fixture_id)
                    events_rows.extend(events)
                    self._stats['total_events'] += len(events)

            # Extract lineups
            lineups_data = self.lineups_loader.load_lineups(season_dir, fixture_id)
            if lineups_data:
                lineups = self.lineup_extractor.extract(lineups_data, fixture_id)
                lineups_rows.extend(lineups)
                self._stats['total_lineups'] += len(lineups)

            # Extract player stats if enabled
            if self.config.include_player_features:
                player_stats = self._extract_player_stats(season_dir, fixture_id)
                if player_stats:
                    players_rows.extend(player_stats)
                    self._stats['total_players'] += len(player_stats)

        # 3. Create DataFrames
        season_data = self._create_dataframes(
            matches_rows, events_rows, players_rows, lineups_rows, teams_map
        )

        # 4. Save season data
        self._save_season_data(season, season_data)

        # Log season summary
        self._log_season_summary(season, season_data)

        return season_data

    def get_statistics(self) -> Dict[str, int]:
        """
        Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        for key in self._stats:
            self._stats[key] = 0
        self.logger.debug("Statistics reset")

    # --- Helper methods ---

    def _extract_player_stats(
            self,
            season_dir: Path,
            fixture_id: int
    ) -> List[Dict[str, Any]]:
        """
        Extract player statistics for a fixture.

        Tries to load player stats from API files. If not available,
        returns empty list.

        Args:
            season_dir: Season directory path
            fixture_id: Fixture identifier

        Returns:
            List of player stat dictionaries
        """
        players_data = self.player_stats_loader.load_player_stats(
            season_dir, fixture_id
        )

        if players_data:
            return self.player_extractor.extract(players_data, fixture_id)

        return []

    def _collect_teams(self, match: Dict[str, Any], teams_map: Dict[int, Dict]) -> None:
        """
        Collect team information from a match.

        Args:
            match: Match dictionary
            teams_map: Dictionary to store unique teams
        """
        if match.get('home_team_id'):
            teams_map[match['home_team_id']] = {
                'team_id': match['home_team_id'],
                'team_name': match.get('home_team_name')
            }

        if match.get('away_team_id'):
            teams_map[match['away_team_id']] = {
                'team_id': match['away_team_id'],
                'team_name': match.get('away_team_name')
            }

    def _create_dataframes(
            self,
            matches_rows: List[Dict],
            events_rows: List[Dict],
            players_rows: List[Dict],
            lineups_rows: List[Dict],
            teams_map: Dict
    ) -> Dict[str, pd.DataFrame]:
        """
        Create DataFrames from extracted data.

        Args:
            matches_rows: List of match dictionaries
            events_rows: List of event dictionaries
            players_rows: List of player stat dictionaries
            lineups_rows: List of lineup dictionaries
            teams_map: Dictionary of teams

        Returns:
            Dictionary with DataFrames
        """
        result = {}

        # Matches
        if matches_rows:
            matches_df = pd.DataFrame(matches_rows)
            matches_df = matches_df.drop_duplicates(subset=['fixture_id'])
            result['matches'] = matches_df
        else:
            result['matches'] = pd.DataFrame()

        # Events
        if events_rows:
            result['events'] = pd.DataFrame(events_rows)
        else:
            result['events'] = pd.DataFrame()

        # Players
        if players_rows:
            result['players'] = pd.DataFrame(players_rows)
        else:
            result['players'] = pd.DataFrame()

        # Lineups
        if lineups_rows:
            result['lineups'] = pd.DataFrame(lineups_rows)
        else:
            result['lineups'] = pd.DataFrame()

        # Teams
        if teams_map:
            result['teams'] = pd.DataFrame(list(teams_map.values()))
        else:
            result['teams'] = pd.DataFrame()

        return result

    def _save_season_data(
            self,
            season: int,
            season_data: Dict[str, pd.DataFrame]
    ) -> None:
        """Save season data to Parquet files."""

        # ZMIANA: Użyj get_output_dir zamiast get_season_dir
        output_dir = self.config.get_output_dir(season)  # ← ZMIENIONE

        if 'matches' in season_data and not season_data['matches'].empty:
            self.writer.write(season_data['matches'], output_dir / "matches.parquet")

        if 'events' in season_data and not season_data['events'].empty:
            self.writer.write(season_data['events'], output_dir / "events.parquet")

        if 'players' in season_data and not season_data['players'].empty:
            self.writer.write(season_data['players'], output_dir / "player_stats.parquet")

        if 'lineups' in season_data and not season_data['lineups'].empty:
            self.writer.write(season_data['lineups'], output_dir / "lineups.parquet")

        if 'teams' in season_data and not season_data['teams'].empty:
            self.writer.write(season_data['teams'], output_dir / "teams.parquet")

    def _combine_seasons(
            self,
            all_matches: List[pd.DataFrame],
            all_events: List[pd.DataFrame],
            all_players: List[pd.DataFrame],
            all_lineups: List[pd.DataFrame],
            all_teams: List[pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Combine DataFrames from multiple seasons.

        Args:
            all_matches: List of match DataFrames
            all_events: List of event DataFrames
            all_players: List of player stat DataFrames
            all_lineups: List of lineup DataFrames
            all_teams: List of team DataFrames

        Returns:
            Dictionary with combined DataFrames
        """
        combined = {}

        if all_matches:
            combined['matches'] = pd.concat(all_matches, ignore_index=True)
            self.logger.info(f"✅ Combined {len(combined['matches'])} matches")
        else:
            combined['matches'] = pd.DataFrame()

        if all_events:
            combined['events'] = pd.concat(all_events, ignore_index=True)
            self.logger.info(f"✅ Combined {len(combined['events'])} events")
        else:
            combined['events'] = pd.DataFrame()

        if all_players:
            combined['players'] = pd.concat(all_players, ignore_index=True)
            self.logger.info(f"✅ Combined {len(combined['players'])} player stats")
        else:
            combined['players'] = pd.DataFrame()

        if all_lineups:
            combined['lineups'] = pd.concat(all_lineups, ignore_index=True)
            self.logger.info(f"✅ Combined {len(combined['lineups'])} lineups")
        else:
            combined['lineups'] = pd.DataFrame()

        if all_teams:
            combined['teams'] = pd.concat(all_teams, ignore_index=True)
            combined['teams'] = combined['teams'].drop_duplicates(subset=['team_id'])
            self.logger.info(f"✅ Combined {len(combined['teams'])} unique teams")
        else:
            combined['teams'] = pd.DataFrame()

        return combined

    def _log_season_summary(self, season: int, season_data: Dict[str, pd.DataFrame]) -> None:
        """
        Log summary for processed season.

        Args:
            season: Season year
            season_data: Dictionary with season DataFrames
        """
        matches_count = len(season_data.get('matches', pd.DataFrame()))
        players_count = len(season_data.get('players', pd.DataFrame()))
        events_count = len(season_data.get('events', pd.DataFrame()))
        lineups_count = len(season_data.get('lineups', pd.DataFrame()))

        self.logger.info(
            f"✅ Season {season} completed: "
            f"{matches_count} matches, "
            f"{players_count} player records, "
            f"{events_count} events, "
            f"{lineups_count} lineups"
        )

    def _log_final_stats(self) -> None:
        """
        Log final processing statistics.

        Args:
            combined: Dictionary with combined DataFrames
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 FINAL PROCESSING STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total matches processed: {self._stats['total_matches']}")
        self.logger.info(f"Total events extracted: {self._stats['total_events']}")
        self.logger.info(f"Total player stats: {self._stats['total_players']}")
        self.logger.info(f"Total lineups: {self._stats['total_lineups']}")

        if self._stats['failed_matches'] > 0:
            self.logger.warning(f"Failed matches: {self._stats['failed_matches']}")

        self.logger.info("=" * 60)
