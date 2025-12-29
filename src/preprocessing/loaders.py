"""Data loading utilities for preprocessing."""
import ast
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from src.preprocessing.exceptions import FileLoadError
from src.preprocessing.interfaces import IDataLoader

logger = logging.getLogger(__name__)


def _extract_fixture_id(value) -> Optional[int]:
    """Extract fixture ID from various formats."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get('id')
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed.get('id')
        except (ValueError, SyntaxError):
            pass
    return None


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


class ParquetDataLoader(IDataLoader):
    """Loader for Parquet files."""

    def load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileLoadError(f"File not found: {path}")
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise FileLoadError(f"Error loading parquet {path}: {e}")


class FixturesLoader:
    """Loader for matches.parquet (flat structure)."""

    def __init__(self, parquet_loader: ParquetDataLoader):
        self.parquet_loader = parquet_loader

    def load_fixtures(self, season_dir: Path) -> List[Dict[str, Any]]:
        """
        Load all fixtures from matches.parquet.
        Returns list of flat dicts with completed matches.
        """
        fixtures_path = season_dir / "matches.parquet"

        try:
            df = self.parquet_loader.load(fixtures_path)

            completed_statuses = {'FT', 'AET', 'PEN'}
            status_col = 'fixture.status.short'

            if status_col in df.columns:
                df_completed = df[df[status_col].isin(completed_statuses)]
            else:
                logger.warning(f"Status column '{status_col}' not found, returning all fixtures")
                df_completed = df

            fixtures = df_completed.to_dict('records')
            logger.info(f"Loaded {len(fixtures)} completed fixtures from {fixtures_path}")
            return fixtures

        except FileLoadError as e:
            logger.error(f"Could not load fixtures: {e}")
            return []


class EventsLoader:
    """Loader for events from aggregated events.parquet."""

    def __init__(self, parquet_loader: ParquetDataLoader):
        self.parquet_loader = parquet_loader

    def load_events(self, season_dir: Path, fixture_id: int) -> Optional[List[Dict[str, Any]]]:
        """Load events for specific match from aggregated parquet."""
        events_file = season_dir / "events.parquet"

        if not events_file.exists():
            logger.debug(f"Events file not found: {events_file}")
            return None

        try:
            df = self.parquet_loader.load(events_file)

            if 'fixture_id' in df.columns:
                df_fixture = df[df['fixture_id'] == fixture_id]
            elif 'fixture.id' in df.columns:
                df_fixture = df[df['fixture.id'] == fixture_id]
            elif 'fixture_info' in df.columns:
                df_fixture = df[df['fixture_info'].apply(_extract_fixture_id) == fixture_id]
            else:
                logger.warning("fixture_id column not found in events.parquet")
                return None

            if df_fixture.empty:
                return None

            return df_fixture.to_dict('records')

        except Exception as e:
            logger.warning(f"Error loading events for fixture {fixture_id}: {e}")
            return None


class LineupsLoader:
    """Loader for lineups from aggregated lineups.parquet."""

    def __init__(self, parquet_loader: ParquetDataLoader):
        self.parquet_loader = parquet_loader

    def load_lineups(self, season_dir: Path, fixture_id: int) -> Optional[List[Dict[str, Any]]]:
        """Load lineups for specific match from aggregated parquet."""
        lineups_file = season_dir / "lineups.parquet"

        if not lineups_file.exists():
            logger.debug(f"Lineups file not found: {lineups_file}")
            return None

        try:
            df = self.parquet_loader.load(lineups_file)

            # Handle different column formats
            if 'fixture_id' in df.columns:
                df_fixture = df[df['fixture_id'] == fixture_id]
            elif 'fixture.id' in df.columns:
                df_fixture = df[df['fixture.id'] == fixture_id]
            elif 'fixture_info' in df.columns:
                df_fixture = df[df['fixture_info'].apply(_extract_fixture_id) == fixture_id]
            else:
                logger.warning("fixture_id column not found in lineups.parquet")
                return None

            if df_fixture.empty:
                return None

            return df_fixture.to_dict('records')

        except Exception as e:
            logger.warning(f"Error loading lineups for fixture {fixture_id}: {e}")
            return None


class PlayerStatsLoader:
    """Loader for player statistics from aggregated player_stats.parquet."""

    def __init__(self, parquet_loader: ParquetDataLoader):
        self.parquet_loader = parquet_loader

    def load_player_stats(self, season_dir: Path, fixture_id: int) -> Optional[List[Dict[str, Any]]]:
        """Load player statistics from aggregated parquet."""
        stats_file = season_dir / "player_stats.parquet"

        if not stats_file.exists():
            logger.debug(f"Player stats file not found: {stats_file}")
            return None

        try:
            df = self.parquet_loader.load(stats_file)

            if 'fixture_id' in df.columns:
                df_fixture = df[df['fixture_id'] == fixture_id]
            elif 'fixture.id' in df.columns:
                df_fixture = df[df['fixture.id'] == fixture_id]
            elif 'fixture_info' in df.columns:
                df_fixture = df[df['fixture_info'].apply(_extract_fixture_id) == fixture_id]
            else:
                logger.warning("fixture_id column not found in player_stats.parquet")
                return None

            if df_fixture.empty:
                return None

            return df_fixture.to_dict('records')

        except Exception as e:
            logger.warning(f"Error loading player stats for fixture {fixture_id}: {e}")
            return None
