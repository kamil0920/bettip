#!/usr/bin/env python3
"""
Fixed Football Data Collector using /fixtures/lineups endpoint
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse
from api_call import FootballAPIClient, APIError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('football_data_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# League configurations
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


class FootballDataManager:
    """Manages football data collection with lineups instead of player statistics."""

    def __init__(self, base_data_dir: str = "football_data"):
        self.base_dir = Path(base_data_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.client = FootballAPIClient()

    def get_season_dir(self, league_key: str, season: int) -> Path:
        """Get or create directory for specific league and season."""
        league_config = LEAGUES_CONFIG[league_key]
        season_dir = self.base_dir / league_config['folder'] / str(season)
        season_dir.mkdir(parents=True, exist_ok=True)
        return season_dir

    def save_json(self, data: Dict, filepath: Path, pretty: bool = True) -> None:
        """Save data to JSON file with metadata."""
        output_data = {
            'metadata': {
                'collected_at': datetime.now().isoformat(),
                'records_count': len(data) if isinstance(data, list) else 1,
                'api_usage': self.client.state.get('count', 0),
                'daily_limit': self.client.daily_limit
            },
            'data': data
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(output_data, f, ensure_ascii=False, default=str)

        logger.info(f"💾 Saved to: {filepath}")

    def load_json(self, filepath: Path) -> Optional[Dict]:
        """Load data from JSON file."""
        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None

    def collect_match_lineups(self, league_key: str, season: int,
                              max_fixtures: int = None, completed_only: bool = True) -> bool:
        """Collect lineups for fixtures in a season."""
        try:
            season_dir = self.get_season_dir(league_key, season)
            fixtures_file = season_dir / 'fixtures.json'

            if not fixtures_file.exists():
                logger.error(f"❌ Fixtures file not found: {fixtures_file}")
                return False

            # Load fixtures
            fixtures_data = self.load_json(fixtures_file)
            if not fixtures_data:
                logger.error("❌ Failed to load fixtures data")
                return False

            fixtures = fixtures_data.get('data', [])

            # Filter fixtures
            if completed_only:
                fixtures = [f for f in fixtures if f['fixture']['status']['short'] == 'FT']

            if len(fixtures) == 0:
                logger.warning(f"⚠️  No completed fixtures found for {league_key} {season}")
                return False

            if max_fixtures:
                fixtures = fixtures[:max_fixtures]

            logger.info(f"👥 Collecting lineups for {len(fixtures)} completed fixtures...")

            # Create lineups subdirectory
            lineups_dir = season_dir / 'lineups'
            lineups_dir.mkdir(exist_ok=True)

            collected_count = 0
            failed_count = 0

            for i, fixture in enumerate(fixtures, 1):
                fixture_id = fixture['fixture']['id']
                lineup_file = lineups_dir / f'fixture_{fixture_id}_lineups.json'

                # Skip if already exists
                if lineup_file.exists():
                    logger.info(f"⏩ {i}/{len(fixtures)}: Fixture {fixture_id} already collected")
                    collected_count += 1
                    continue

                try:
                    home_team = fixture['teams']['home']['name']
                    away_team = fixture['teams']['away']['name']
                    fixture_date = fixture['fixture']['date']

                    logger.info(f"📥 {i}/{len(fixtures)}: {home_team} vs {away_team}")
                    logger.info(f"   Date: {fixture_date}, ID: {fixture_id}")

                    # Get lineups using the working endpoint
                    lineups = self.client._make_request('/fixtures/lineups', {'fixture': fixture_id})
                    lineup_data = lineups.get('response', [])

                    if lineup_data and len(lineup_data) > 0:
                        # Count total players
                        total_players = 0
                        for team_lineup in lineup_data:
                            if 'startXI' in team_lineup:
                                total_players += len(team_lineup['startXI'])
                            if 'substitutes' in team_lineup:
                                total_players += len(team_lineup['substitutes'])

                        # Add fixture metadata
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

                        logger.info(f"   ✅ Collected {total_players} total players from {len(lineup_data)} teams")
                    else:
                        logger.warning(f"   ⚠️  No lineup data for fixture {fixture_id}")
                        failed_count += 1

                except APIError as e:
                    logger.error(f"   ❌ API Error for fixture {fixture_id}: {e}")
                    failed_count += 1
                    continue
                except Exception as e:
                    logger.error(f"   ❌ Unexpected error for fixture {fixture_id}: {e}")
                    failed_count += 1
                    continue

            logger.info(f"📊 Lineups collection summary:")
            logger.info(f"   ✅ Successfully collected: {collected_count}")
            logger.info(f"   ❌ Failed: {failed_count}")
            logger.info(f"   📊 Success rate: {collected_count / (collected_count + failed_count) * 100:.1f}%" if (
                                                                                                                             collected_count + failed_count) > 0 else "N/A")

            return collected_count > 0

        except Exception as e:
            logger.error(f"❌ Failed to collect lineups: {e}")
            return False

    def collect_match_events(self, league_key: str, season: int,
                             max_fixtures: int = None, completed_only: bool = True) -> bool:
        """Collect match events (goals, cards, substitutions) for fixtures."""
        try:
            season_dir = self.get_season_dir(league_key, season)
            fixtures_file = season_dir / 'fixtures.json'

            if not fixtures_file.exists():
                logger.error(f"❌ Fixtures file not found: {fixtures_file}")
                return False

            fixtures_data = self.load_json(fixtures_file)
            if not fixtures_data:
                return False

            fixtures = fixtures_data.get('data', [])

            if completed_only:
                fixtures = [f for f in fixtures if f['fixture']['status']['short'] == 'FT']

            if len(fixtures) == 0:
                logger.warning(f"⚠️  No completed fixtures found for events collection")
                return False

            if max_fixtures:
                fixtures = fixtures[:max_fixtures]

            logger.info(f"⚽ Collecting events for {len(fixtures)} fixtures...")

            # Create events subdirectory
            events_dir = season_dir / 'events'
            events_dir.mkdir(exist_ok=True)

            collected_count = 0

            for i, fixture in enumerate(fixtures, 1):
                fixture_id = fixture['fixture']['id']
                events_file = events_dir / f'fixture_{fixture_id}_events.json'

                if events_file.exists():
                    logger.info(f"⏩ {i}/{len(fixtures)}: Events for fixture {fixture_id} already collected")
                    collected_count += 1
                    continue

                try:
                    home_team = fixture['teams']['home']['name']
                    away_team = fixture['teams']['away']['name']

                    logger.info(f"📥 {i}/{len(fixtures)}: Getting events for {home_team} vs {away_team}")

                    # Get events
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

                        logger.info(f"   ✅ Collected {len(events_data)} events")
                    else:
                        logger.info(f"   ℹ️  No events data for fixture {fixture_id}")

                except APIError as e:
                    logger.error(f"   ❌ API Error for events {fixture_id}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"   ❌ Unexpected error for events {fixture_id}: {e}")
                    continue

            logger.info(f"📊 Events collection: {collected_count} fixtures processed")
            return collected_count > 0

        except Exception as e:
            logger.error(f"❌ Failed to collect events: {e}")
            return False

    def collect_season_data(self, league_key: str, season: int, force_refresh: bool = False) -> bool:
        """Collect complete data for a specific league and season."""
        try:
            league_config = LEAGUES_CONFIG[league_key]
            season_dir = self.get_season_dir(league_key, season)

            logger.info(f"🏆 Collecting {league_config['name']} data for season {season}")
            logger.info(f"📁 Directory: {season_dir}")

            # Files to collect
            files_to_collect = [
                ('teams.json', '/teams', {'league': league_config['id'], 'season': season}),
                ('fixtures.json', '/fixtures', {'league': league_config['id'], 'season': season}),
                ('standings.json', '/standings', {'league': league_config['id'], 'season': season}),
            ]

            collected_files = []

            for filename, endpoint, params in files_to_collect:
                filepath = season_dir / filename

                # Skip if file exists and not forcing refresh
                if filepath.exists() and not force_refresh:
                    logger.info(f"⏩ Skipping {filename} (already exists)")
                    collected_files.append(filename)
                    continue

                try:
                    logger.info(f"📥 Fetching {filename}...")

                    if endpoint == '/fixtures':
                        # Use the client's method for fixtures
                        data = self.client.get_fixtures(league_config['id'], season)
                    else:
                        # Use generic request for other endpoints
                        response = self.client._make_request(endpoint, params)
                        data = response.get('response', [])

                    self.save_json(data, filepath)
                    collected_files.append(filename)

                    logger.info(f"✅ {filename}: {len(data) if isinstance(data, list) else 1} records")

                except APIError as e:
                    logger.error(f"❌ Failed to collect {filename}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Unexpected error collecting {filename}: {e}")
                    continue

            # Create summary file
            summary = {
                'league': league_config,
                'season': season,
                'collected_files': collected_files,
                'collection_date': datetime.now().isoformat(),
                'api_usage': self.client.state.get('count', 0)
            }

            self.save_json(summary, season_dir / 'collection_summary.json')

            logger.info(f"📊 Season {season} summary: {len(collected_files)} files collected")
            return len(collected_files) > 0

        except Exception as e:
            logger.error(f"❌ Failed to collect season data: {e}")
            return False


def bulk_collect_with_lineups(league_key: str = 'premier_league',
                              start_season: int = 2020,
                              end_season: int = None,
                              include_lineups: bool = False,
                              include_events: bool = False,
                              max_fixtures_per_season: int = None):
    """Collect historical data with lineups and events."""
    if end_season is None:
        end_season = datetime.now().year

    manager = FootballDataManager()
    current_year = datetime.now().year

    logger.info(f"🚀 Starting bulk collection for {league_key}")
    logger.info(f"📊 Seasons: {start_season} to {end_season}")
    logger.info(f"👥 Include lineups: {include_lineups}")
    logger.info(f"⚽ Include events: {include_events}")
    logger.info(f"📅 Current year: {current_year}")

    total_seasons = end_season - start_season + 1
    successful_seasons = 0

    for season in range(start_season, end_season + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"📅 Processing season {season} ({season - start_season + 1}/{total_seasons})")
        logger.info(f"📊 API Usage: {manager.client.state.get('count', 0)}/{manager.client.daily_limit}")

        # Check if we're getting close to daily limit
        remaining = manager.client.daily_limit - manager.client.state.get('count', 0)
        if remaining < 100:  # Conservative threshold
            logger.warning(f"⚠️  Approaching daily limit. Remaining: {remaining}")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                break

        # Collect basic season data
        success = manager.collect_season_data(league_key, season)

        if success:
            successful_seasons += 1

            # Collect lineups if requested and season has completed matches
            if include_lineups:
                logger.info(f"👥 Collecting lineups for season {season}...")
                manager.collect_match_lineups(
                    league_key, season,
                    max_fixtures=max_fixtures_per_season,
                    completed_only=True
                )

            # Collect events if requested
            if include_events:
                logger.info(f"⚽ Collecting events for season {season}...")
                manager.collect_match_events(
                    league_key, season,
                    max_fixtures=max_fixtures_per_season,
                    completed_only=True
                )
        else:
            logger.error(f"❌ Failed to collect data for season {season}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"📊 Bulk collection completed!")
    logger.info(f"✅ Successful seasons: {successful_seasons}/{total_seasons}")
    logger.info(f"📊 Final API usage: {manager.client.state.get('count', 0)}/{manager.client.daily_limit}")


def main():
    """Main entry point with updated arguments."""
    parser = argparse.ArgumentParser(description='Football Data Collector with Lineups')
    parser.add_argument('--mode', choices=['bulk', 'season'],
                        default='bulk', help='Collection mode')
    parser.add_argument('--league', choices=list(LEAGUES_CONFIG.keys()),
                        default='premier_league', help='League to collect')
    parser.add_argument('--start-season', type=int, default=2020,
                        help='Start season for bulk collection')
    parser.add_argument('--end-season', type=int,
                        help='End season for bulk collection (default: current year)')
    parser.add_argument('--season', type=int,
                        help='Specific season for single season collection')
    parser.add_argument('--include-lineups', action='store_true',
                        help='Include match lineups (team formations and player lists)')
    parser.add_argument('--include-events', action='store_true',
                        help='Include match events (goals, cards, substitutions)')
    parser.add_argument('--max-fixtures', type=int,
                        help='Maximum fixtures per season (for testing)')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh existing files')

    args = parser.parse_args()

    if args.mode == 'bulk':
        bulk_collect_with_lineups(
            league_key=args.league,
            start_season=args.start_season,
            end_season=args.end_season,
            include_lineups=args.include_lineups,
            include_events=args.include_events,
            max_fixtures_per_season=args.max_fixtures
        )
    elif args.mode == 'season':
        if not args.season:
            args.season = datetime.now().year

        manager = FootballDataManager()
        manager.collect_season_data(args.league, args.season, args.force_refresh)

        if args.include_lineups:
            manager.collect_match_lineups(args.league, args.season, args.max_fixtures)

        if args.include_events:
            manager.collect_match_events(args.league, args.season, args.max_fixtures)


if __name__ == "__main__":
    main()
