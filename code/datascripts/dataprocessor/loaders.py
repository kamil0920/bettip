import json
import logging
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any, List

from exceptions import FileLoadError
from interfaces import IDataLoader

logger = logging.getLogger(__name__)

class JSONDataLoader(IDataLoader):
    """Loader for JSON files."""

    def load(self, path: Path) -> Dict[str, Any]:
        """Load data from json file."""
        if not path.exists():
            raise FileLoadError(f"File not found: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise FileLoadError(f"Invalid JSON in {path}: {e}")
        except Exception as e:
            raise FileLoadError(f"Error loading {path}: {e}")


class FixturesLoader:
    """Loader dla fixtures.json."""

    def __init__(self, json_loader: JSONDataLoader):
        self.json_loader = json_loader

    def load_fixtures(self, season_dir: Path) -> List[Dict[str, Any]]:
        """
        Load all fixtures from fixtures.json.
        Return list fixtures files ready to process.
        """
        fixtures_path = season_dir / "fixtures.json"

        try:
            data = self.json_loader.load(fixtures_path)
            fixtures = data.get('data', [])

            completed_statuses = {'FT', 'AET', 'PEN'}
            filtered_fixtures = [
                f for f in fixtures
                if f.get('fixture', {}).get('status', {}).get('short') in completed_statuses
            ]

            logger.info(f"Loaded {len(filtered_fixtures)} completed fixtures from {fixtures_path}")
            return filtered_fixtures

        except FileLoadError as e:
            logger.error(f"Could not load fixtures: {e}")
            return []


class EventsLoader:
    """Loader for events from /events/ directory."""

    def __init__(self, json_loader: JSONDataLoader):
        self.json_loader = json_loader

    def load_events(self, season_dir: Path, fixture_id: int) -> Optional[List[Dict[str, Any]]]:
        """Load events for specific match."""
        events_dir = season_dir / "events"
        events_file = events_dir / f"fixture_{fixture_id}_events.json"

        if not events_file.exists():
            logger.debug(f"Events file not found: {events_file}")
            return None

        try:
            data = self.json_loader.load(events_file)
            return data.get('data', {}).get('events', [])
        except Exception as e:
            logger.warning(f"Error loading events for fixture {fixture_id}: {e}")
            return None


class LineupsLoader:
    """Loader for lineups from /lineups/ directory."""

    def __init__(self, json_loader: JSONDataLoader):
        self.json_loader = json_loader

    def load_lineups(self, season_dir: Path, fixture_id: int) -> Optional[List[Dict[str, Any]]]:
        """Load lineups for specific match."""
        lineups_dir = season_dir / "lineups"
        lineups_file = lineups_dir / f"fixture_{fixture_id}_lineups.json"

        if not lineups_file.exists():
            logger.debug(f"Lineups file not found: {lineups_file}")
            return None

        try:
            data = self.json_loader.load(lineups_file)
            return data.get('data', {}).get('lineups', [])
        except Exception as e:
            logger.warning(f"Error loading lineups for fixture {fixture_id}: {e}")
            return None


class PlayerStatsLoader:
    """Loader for player statistics from /players/ directory."""

    def __init__(self, json_loader: JSONDataLoader):
        self.json_loader = json_loader

    def load_player_stats(self, season_dir: Path, fixture_id: int) -> Optional[List[Dict[str, Any]]]:
        """
        Load player statistics from API (full stats).
        """
        players_dir = season_dir / "players"
        players_file = players_dir / f"fixture_{fixture_id}_players.json"

        if not players_file.exists():
            logger.debug(f"Player stats file not found: {players_file}")
            return None

        try:
            data = self.json_loader.load(players_file)

            if 'data' not in data or 'players' not in data['data']:
                return None

            return data['data']['players']

        except Exception as e:
            logger.warning(f"Error loading player stats for fixture {fixture_id}: {e}")
            return None