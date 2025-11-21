"""
Football data collector for fetching and storing API data.

Manages collection of fixtures, lineups, events, and player statistics
from the Football API.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from api_client import FootballAPIClient, APIError

logger = logging.getLogger(__name__)

LEAGUES_CONFIG = {
    'premier_league': {
        'id': 39,
        'name': 'Premier League',
        'country': 'England',
        'folder': 'premier_league'
    },
    'la_liga': {
        'id': 140,
        'name': 'La Liga',
        'country': 'Spain',
        'folder': 'la_liga'
    },
    'bundesliga': {
        'id': 78,
        'name': 'Bundesliga',
        'country': 'Germany',
        'folder': 'bundesliga'
    },
    'serie_a': {
        'id': 135,
        'name': 'Serie A',
        'country': 'Italy',
        'folder': 'serie_a'
    },
    'ligue_1': {
        'id': 61,
        'name': 'Ligue 1',
        'country': 'France',
        'folder': 'ligue_1'
    }
}


class FootballDataCollector:
    """Manages football data collection with lineups, events, and player statistics."""

    def __init__(self, base_data_dir: str = "data/01-raw"):
        """
        Initialize the data collector.

        Args:
            base_data_dir: Base directory for storing raw data
        """
        self.base_dir = Path(base_data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.client = FootballAPIClient()
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_season_dir(self, league_key: str, season: int) -> Path:
        """
        Get or create directory for specific league and season.

        Args:
            league_key: League identifier (e.g., 'premier_league')
            season: Season year

        Returns:
            Path to season directory
        """
        league_config = LEAGUES_CONFIG[league_key]
        season_dir = self.base_dir / league_config['folder'] / str(season)
        season_dir.mkdir(parents=True, exist_ok=True)
        return season_dir

    def save_json(self, data: Any, filepath: Path, pretty: bool = True) -> None:
        """
        Save data to JSON file with metadata.

        Args:
            data: Data to save
            filepath: Output file path
            pretty: Whether to format JSON with indentation
        """
        output_data = {
            'metadata': {
                'collected_at': datetime.now().isoformat(),
                'records_count': len(data) if isinstance(data, list) else 1,
                'api_usage': self.client.state.get('count', 0),
                'daily_limit': self.client.daily_limit
            },
            'data': data
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(output_data, f, ensure_ascii=False, default=str)

        self.logger.info(f"Saved to: {filepath}")

    def load_json(self, filepath: Path) -> Optional[Dict]:
        """
        Load data from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Loaded data or None if file doesn't exist
        """
        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load {filepath}: {e}")
            return None

    def collect_season_data(self, league_key: str, season: int, force_refresh: bool = False) -> bool:
        """
        Collect complete data for a specific league and season.

        This collects teams, fixtures, and standings for the season.

        Args:
            league_key: League identifier
            season: Season year
            force_refresh: Whether to overwrite existing files

        Returns:
            True if collection was successful
        """
        try:
            league_config = LEAGUES_CONFIG[league_key]
            season_dir = self.get_season_dir(league_key, season)

            self.logger.info(f"Collecting {league_config['name']} data for season {season}")
            self.logger.info(f"Directory: {season_dir}")

            files_to_collect = [
                ('teams.json', '/teams', {'league': league_config['id'], 'season': season}),
                ('fixtures.json', '/fixtures', {'league': league_config['id'], 'season': season}),
                ('standings.json', '/standings', {'league': league_config['id'], 'season': season}),
            ]

            collected_files = []

            for filename, endpoint, params in files_to_collect:
                filepath = season_dir / filename

                if filepath.exists() and not force_refresh:
                    self.logger.info(f"Skipping {filename} (already exists)")
                    collected_files.append(filename)
                    continue

                try:
                    self.logger.info(f"Fetching {filename}...")

                    if endpoint == '/fixtures':
                        data = self.client.get_fixtures(league_config['id'], season)
                    else:
                        response = self.client._make_request(endpoint, params)
                        data = response.get('response', [])

                    self.save_json(data, filepath)
                    collected_files.append(filename)

                    self.logger.info(f"{filename}: {len(data) if isinstance(data, list) else 1} records")

                except APIError as e:
                    self.logger.error(f"Failed to collect {filename}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error collecting {filename}: {e}")
                    continue

            summary = {
                'league': league_config,
                'season': season,
                'collected_files': collected_files,
                'collection_date': datetime.now().isoformat(),
                'api_usage': self.client.state.get('count', 0)
            }

            self.save_json(summary, season_dir / 'collection_summary.json')

            self.logger.info(f"Season {season} summary: {len(collected_files)} files collected")
            return len(collected_files) > 0

        except Exception as e:
            self.logger.error(f"Failed to collect season data: {e}")
            return False

    def collect_match_lineups(
            self,
            league_key: str,
            season: int,
            max_fixtures: Optional[int] = None,
            completed_only: bool = True
    ) -> bool:
        """
        Collect lineups for fixtures in a season.

        Args:
            league_key: League identifier
            season: Season year
            max_fixtures: Maximum number of fixtures to process
            completed_only: Only process completed fixtures

        Returns:
            True if at least one lineup was collected
        """
        try:
            season_dir = self.get_season_dir(league_key, season)
            fixtures_file = season_dir / 'fixtures.json'

            if not fixtures_file.exists():
                self.logger.error(f"Fixtures file not found: {fixtures_file}")
                return False

            fixtures_data = self.load_json(fixtures_file)
            if not fixtures_data:
                self.logger.error("Failed to load fixtures data")
                return False

            fixtures = fixtures_data.get('data', [])

            if completed_only:
                fixtures = [f for f in fixtures if f['fixture']['status']['short'] == 'FT']

            if len(fixtures) == 0:
                self.logger.warning(f"No completed fixtures found for {league_key} {season}")
                return False

            if max_fixtures:
                fixtures = fixtures[:max_fixtures]

            self.logger.info(f"Collecting lineups for {len(fixtures)} completed fixtures...")

            lineups_dir = season_dir / 'lineups'
            lineups_dir.mkdir(exist_ok=True)

            collected_count = 0
            failed_count = 0

            for i, fixture in enumerate(fixtures, 1):
                fixture_id = fixture['fixture']['id']
                lineup_file = lineups_dir / f'fixture_{fixture_id}_lineups.json'

                if lineup_file.exists():
                    self.logger.info(f"{i}/{len(fixtures)}: Fixture {fixture_id} already collected")
                    collected_count += 1
                    continue

                try:
                    home_team = fixture['teams']['home']['name']
                    away_team = fixture['teams']['away']['name']
                    fixture_date = fixture['fixture']['date']

                    self.logger.info(f"{i}/{len(fixtures)}: {home_team} vs {away_team}")

                    lineups = self.client._make_request('/fixtures/lineups', {'fixture': fixture_id})
                    lineup_data = lineups.get('response', [])

                    if lineup_data and len(lineup_data) > 0:
                        total_players = sum(
                            len(team_lineup.get('startXI', [])) + len(team_lineup.get('substitutes', []))
                            for team_lineup in lineup_data
                        )

                        lineup_package = {
                            'fixture_info': {
                                'id': fixture_id,
                                'date': fixture_date,
                                'home_team': home_team,
                                'away_team': away_team,
                                'score': fixture.get('goals', {}),
                                'status': fixture['fixture']['status']['short']
                            },
                            'lineups': lineup_data
                        }

                        self.save_json(lineup_package, lineup_file)
                        collected_count += 1

                        self.logger.info(f"Collected {total_players} total players from {len(lineup_data)} teams")
                    else:
                        self.logger.warning(f"No lineup data for fixture {fixture_id}")
                        failed_count += 1

                except APIError as e:
                    self.logger.error(f"API Error for fixture {fixture_id}: {e}")
                    failed_count += 1
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error for fixture {fixture_id}: {e}")
                    failed_count += 1
                    continue

            self._log_collection_summary("Lineups", collected_count, failed_count)
            return collected_count > 0

        except Exception as e:
            self.logger.error(f"Failed to collect lineups: {e}")
            return False

    def collect_match_events(
            self,
            league_key: str,
            season: int,
            max_fixtures: Optional[int] = None,
            completed_only: bool = True
    ) -> bool:
        """
        Collect match events (goals, cards, substitutions) for fixtures.

        Args:
            league_key: League identifier
            season: Season year
            max_fixtures: Maximum number of fixtures to process
            completed_only: Only process completed fixtures

        Returns:
            True if at least one event set was collected
        """
        try:
            season_dir = self.get_season_dir(league_key, season)
            fixtures_file = season_dir / 'fixtures.json'

            if not fixtures_file.exists():
                self.logger.error(f"Fixtures file not found: {fixtures_file}")
                return False

            fixtures_data = self.load_json(fixtures_file)
            if not fixtures_data:
                return False

            fixtures = fixtures_data.get('data', [])

            if completed_only:
                fixtures = [f for f in fixtures if f['fixture']['status']['short'] == 'FT']

            if len(fixtures) == 0:
                self.logger.warning("No completed fixtures found for events collection")
                return False

            if max_fixtures:
                fixtures = fixtures[:max_fixtures]

            self.logger.info(f"Collecting events for {len(fixtures)} fixtures...")

            events_dir = season_dir / 'events'
            events_dir.mkdir(exist_ok=True)

            collected_count = 0

            for i, fixture in enumerate(fixtures, 1):
                fixture_id = fixture['fixture']['id']
                events_file = events_dir / f'fixture_{fixture_id}_events.json'

                if events_file.exists():
                    self.logger.info(f"{i}/{len(fixtures)}: Events for fixture {fixture_id} already collected")
                    collected_count += 1
                    continue

                try:
                    home_team = fixture['teams']['home']['name']
                    away_team = fixture['teams']['away']['name']

                    self.logger.info(f"{i}/{len(fixtures)}: Getting events for {home_team} vs {away_team}")

                    events_response = self.client._make_request('/fixtures/events', {'fixture': fixture_id})
                    events_data = events_response.get('response', [])

                    if events_data:
                        events_package = {
                            'fixture_info': {
                                'id': fixture_id,
                                'date': fixture['fixture']['date'],
                                'home_team': home_team,
                                'away_team': away_team,
                                'score': fixture.get('goals', {})
                            },
                            'events': events_data
                        }

                        self.save_json(events_package, events_file)
                        collected_count += 1

                        self.logger.info(f"Collected {len(events_data)} events")
                    else:
                        self.logger.info(f"No events data for fixture {fixture_id}")

                except APIError as e:
                    self.logger.error(f"API Error for events {fixture_id}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error for events {fixture_id}: {e}")
                    continue

            self.logger.info(f"Events collection: {collected_count} fixtures processed")
            return collected_count > 0

        except Exception as e:
            self.logger.error(f"Failed to collect events: {e}")
            return False

    def collect_player_statistics(
            self,
            league_key: str,
            season: int,
            max_fixtures: Optional[int] = None,
            completed_only: bool = True
    ) -> bool:
        """
        Collect player statistics (rating, shots, passes, etc.) for fixtures.

        Args:
            league_key: League identifier
            season: Season year
            max_fixtures: Maximum number of fixtures to process
            completed_only: Only process completed fixtures

        Returns:
            True if at least one set of player stats was collected
        """
        try:
            season_dir = self.get_season_dir(league_key, season)
            fixtures_file = season_dir / 'fixtures.json'

            if not fixtures_file.exists():
                self.logger.error(f"Fixtures file not found: {fixtures_file}")
                return False

            fixtures_data = self.load_json(fixtures_file)
            if not fixtures_data:
                return False

            fixtures = fixtures_data.get('data', [])

            if completed_only:
                fixtures = [f for f in fixtures if f['fixture']['status']['short'] == 'FT']

            if len(fixtures) == 0:
                self.logger.warning("No completed fixtures found for player statistics")
                return False

            if max_fixtures:
                fixtures = fixtures[:max_fixtures]

            self.logger.info(f"Collecting player statistics for {len(fixtures)} fixtures...")

            players_dir = season_dir / 'players'
            players_dir.mkdir(exist_ok=True)

            collected_count = 0
            failed_count = 0

            for i, fixture in enumerate(fixtures, 1):
                fixture_id = fixture['fixture']['id']
                players_file = players_dir / f'fixture_{fixture_id}_players.json'

                if players_file.exists():
                    self.logger.info(f"{i}/{len(fixtures)}: Player stats for fixture {fixture_id} already collected")
                    collected_count += 1
                    continue

                try:
                    home_team = fixture['teams']['home']['name']
                    away_team = fixture['teams']['away']['name']
                    fixture_date = fixture['fixture']['date']

                    self.logger.info(f"{i}/{len(fixtures)}: {home_team} vs {away_team}")

                    players_response = self.client._make_request('/fixtures/players', {'fixture': fixture_id})
                    players_data = players_response.get('response', [])

                    if players_data and len(players_data) > 0:
                        total_players = sum(
                            len(team.get('players', []))
                            for team in players_data
                        )

                        players_package = {
                            'fixture_info': {
                                'id': fixture_id,
                                'date': fixture_date,
                                'home_team': home_team,
                                'away_team': away_team,
                                'score': fixture.get('goals', {}),
                                'status': fixture['fixture']['status']['short']
                            },
                            'players': players_data
                        }

                        self.save_json(players_package, players_file)
                        collected_count += 1

                        self.logger.info(
                            f"Collected statistics for {total_players} players from {len(players_data)} teams"
                        )
                    else:
                        self.logger.warning(f"No player statistics for fixture {fixture_id}")
                        failed_count += 1

                except APIError as e:
                    self.logger.error(f"API Error for fixture {fixture_id}: {e}")
                    failed_count += 1
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error for fixture {fixture_id}: {e}")
                    failed_count += 1
                    continue

            self._log_collection_summary("Player statistics", collected_count, failed_count)
            return collected_count > 0

        except Exception as e:
            self.logger.error(f"Failed to collect player statistics: {e}")
            return False

    def _log_collection_summary(self, data_type: str, collected: int, failed: int) -> None:
        """Log collection summary statistics."""
        total = collected + failed
        self.logger.info(f"{data_type} collection summary:")
        self.logger.info(f"  Successfully collected: {collected}")
        self.logger.info(f"  Failed: {failed}")
        if total > 0:
            self.logger.info(f"  Success rate: {collected / total * 100:.1f}%")


def bulk_collect(
        league_key: str = 'premier_league',
        start_season: int = 2020,
        end_season: Optional[int] = None,
        include_lineups: bool = False,
        include_events: bool = False,
        include_player_stats: bool = False,
        max_fixtures_per_season: Optional[int] = None,
        base_data_dir: str = "data/01-raw"
) -> None:
    """
    Collect historical data with lineups, events, and player statistics.

    Args:
        league_key: League identifier
        start_season: First season to collect
        end_season: Last season to collect (defaults to current year)
        include_lineups: Whether to collect lineups
        include_events: Whether to collect events
        include_player_stats: Whether to collect player statistics
        max_fixtures_per_season: Limit fixtures per season (for testing)
        base_data_dir: Base directory for raw data
    """
    if end_season is None:
        end_season = datetime.now().year

    collector = FootballDataCollector(base_data_dir)

    logger.info(f"Starting bulk collection for {league_key}")
    logger.info(f"Seasons: {start_season} to {end_season}")
    logger.info(f"Include lineups: {include_lineups}")
    logger.info(f"Include events: {include_events}")
    logger.info(f"Include player stats: {include_player_stats}")

    total_seasons = end_season - start_season + 1
    successful_seasons = 0

    for season in range(start_season, end_season + 1):
        logger.info("=" * 60)
        logger.info(f"Processing season {season} ({season - start_season + 1}/{total_seasons})")
        logger.info(f"API Usage: {collector.client.state.get('count', 0)}/{collector.client.daily_limit}")

        remaining = collector.client.daily_limit - collector.client.state.get('count', 0)
        if remaining < 100:
            logger.warning(f"Approaching daily limit. Remaining: {remaining}")
            break

        success = collector.collect_season_data(league_key, season)

        if success:
            successful_seasons += 1

            if include_lineups:
                logger.info(f"Collecting lineups for season {season}...")
                collector.collect_match_lineups(
                    league_key, season,
                    max_fixtures=max_fixtures_per_season,
                    completed_only=True
                )

            if include_events:
                logger.info(f"Collecting events for season {season}...")
                collector.collect_match_events(
                    league_key, season,
                    max_fixtures=max_fixtures_per_season,
                    completed_only=True
                )

            if include_player_stats:
                logger.info(f"Collecting player statistics for season {season}...")
                collector.collect_player_statistics(
                    league_key, season,
                    max_fixtures=max_fixtures_per_season,
                    completed_only=True
                )
        else:
            logger.error(f"Failed to collect data for season {season}")

    logger.info("=" * 60)
    logger.info("Bulk collection completed!")
    logger.info(f"Successful seasons: {successful_seasons}/{total_seasons}")
    logger.info(f"Final API usage: {collector.client.state.get('count', 0)}/{collector.client.daily_limit}")
