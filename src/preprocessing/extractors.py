"""Data extraction implementations for preprocessing."""
import logging
from typing import Dict, Any, List, Optional

import pandas as pd

from src.preprocessing.interfaces import (
    IFixtureExtractor,
    IDataValidator,
    IEventExtractor,
    IPlayerStatsExtractor,
    ILineupExtractor
)

logger = logging.getLogger(__name__)


class NestedDataHelper:
    """Helper for working with nested and flat structures."""

    @staticmethod
    def get_nested(obj: Any, *keys: str) -> Optional[Any]:
        """Get nested value from dict."""
        current = obj
        for key in keys:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
        return current

    @staticmethod
    def get_flat(obj: Dict, *keys: str) -> Optional[Any]:
        """
        Get value using flat key (dot notation).
        Example: get_flat(obj, 'fixture', 'id') -> obj.get('fixture.id')
        Falls back to nested access if flat key not found.
        """
        if not obj:
            return None

        flat_key = '.'.join(keys)
        if flat_key in obj:
            return obj[flat_key]

        current = obj
        for key in keys:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
        return current

    @staticmethod
    def safe_int(value: Any) -> Optional[int]:
        """Safe conversion to int."""
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
            return int(float(value))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def safe_float(value: Any) -> Optional[float]:
        """Safe conversion to float."""
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (ValueError, TypeError):
            return None


class FixtureExtractor(IFixtureExtractor):
    """
    Fixture extractor implementation.
    Supports both flat (parquet) and nested (JSON) data formats.
    """

    def __init__(self, validator: IDataValidator):
        self.validator = validator
        self.helper = NestedDataHelper()

    def extract(self, fixture_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract single fixture data."""
        try:
            extracted = self._extract_impl(fixture_data)

            if extracted and self.validator.validate(extracted):
                return extracted

            if extracted:
                logger.debug(f"Validation failed: {self.validator.get_errors()}")

            return None

        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return None

    def _extract_impl(self, fixture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of extraction logic - supports flat keys."""
        get = self.helper.get_flat

        return {
            "fixture_id": self.helper.safe_int(get(fixture_data, "fixture", "id")),
            "date": get(fixture_data, "fixture", "date"),
            "timestamp": self.helper.safe_int(get(fixture_data, "fixture", "timestamp")),
            "referee": get(fixture_data, "fixture", "referee"),
            "venue_id": self.helper.safe_int(get(fixture_data, "fixture", "venue", "id")),
            "venue_name": get(fixture_data, "fixture", "venue", "name"),
            "status": get(fixture_data, "fixture", "status", "short"),
            "league_id": self.helper.safe_int(get(fixture_data, "league", "id")),
            "round": get(fixture_data, "league", "round"),
            "home_team_id": self.helper.safe_int(get(fixture_data, "teams", "home", "id")),
            "home_team_name": get(fixture_data, "teams", "home", "name"),
            "away_team_id": self.helper.safe_int(get(fixture_data, "teams", "away", "id")),
            "away_team_name": get(fixture_data, "teams", "away", "name"),
            "ft_home": self.helper.safe_int(get(fixture_data, "score", "fulltime", "home")),
            "ft_away": self.helper.safe_int(get(fixture_data, "score", "fulltime", "away")),
            "ht_home": self.helper.safe_int(get(fixture_data, "score", "halftime", "home")),
            "ht_away": self.helper.safe_int(get(fixture_data, "score", "halftime", "away")),
        }


class EventExtractor(IEventExtractor):
    """
    Event extractor with validation.
    Supports flat parquet data where each row is already a single event.
    """

    def __init__(self, validator: IDataValidator):
        self.validator = validator
        self.helper = NestedDataHelper()

    def extract(self, events_data: List[Dict[str, Any]], fixture_id: int) -> List[Dict[str, Any]]:
        """
        Extract events with validation.
        Events data is now flat - each dict is already a single event.
        """
        extracted_events = []
        invalid_count = 0

        for idx, event in enumerate(events_data):
            try:
                event_type = event.get('type') or self.helper.get_flat(event, 'type')

                if event_type in ['Goal', 'Card']:
                    event_row = self._extract_single_event(event, fixture_id)

                    if event_row is None:
                        continue

                    if self.validator.validate(event_row):
                        extracted_events.append(event_row)
                    else:
                        invalid_count += 1
                        logger.debug(
                            f"Event {idx} validation failed: {self.validator.get_errors()}"
                        )

            except Exception as e:
                logger.warning(f"Error extracting event {idx}: {e}")
                continue

        if invalid_count > 0:
            logger.warning(
                f"Fixture {fixture_id}: {invalid_count} events failed validation"
            )

        return extracted_events

    def _extract_single_event(self, event: Dict[str, Any], fixture_id: int) -> Optional[Dict[str, Any]]:
        """Extract single event data - supports flat keys."""
        get = self.helper.get_flat

        return {
            'fixture_id': event.get('fixture_id') or fixture_id,
            'type': event.get('type'),
            'detail': event.get('detail'),
            'time_elapsed': get(event, 'time', 'elapsed'),
            'team_id': get(event, 'team', 'id'),
            'player_id': get(event, 'player', 'id'),
            'player_name': get(event, 'player', 'name'),
            'assist_id': get(event, 'assist', 'id'),
            'assist_name': get(event, 'assist', 'name'),
        }


class PlayerStatsExtractor(IPlayerStatsExtractor):
    """
    Player statistics extractor implementation.
    Supports flat parquet data where each row is already a single player's stats.
    """

    def __init__(self, validator: IDataValidator):
        self.validator = validator
        self.helper = NestedDataHelper()

    def extract(self, players_data: List[Dict[str, Any]], fixture_id: int) -> List[Dict[str, Any]]:
        """
        Extract player statistics.
        Data is now flat - each dict is already one player's stats for one match.
        """
        extracted_stats = []

        for player in players_data:
            player_stat = self._extract_single_player_flat(player, fixture_id)

            if player_stat and self.validator.validate(player_stat):
                extracted_stats.append(player_stat)
            else:
                if player_stat:
                    logger.debug(f"Player stat validation failed: {self.validator.get_errors()}")

        return extracted_stats

    def _extract_single_player_flat(self, player: Dict[str, Any], fixture_id: int) -> Optional[Dict[str, Any]]:
        """Extract single player data from flat parquet row."""
        try:
            get = self.helper.get_flat
            safe_int = self.helper.safe_int
            safe_float = self.helper.safe_float

            return {
                'fixture_id': player.get('fixture_id') or fixture_id,
                'player_id': safe_int(player.get('id') or get(player, 'player', 'id')),
                'player_name': player.get('name') or get(player, 'player', 'name'),
                'team_id': safe_int(get(player, 'team', 'id')),
                'team_name': player.get('team_name'),

                'minutes': safe_int(get(player, 'games', 'minutes')) or 0,
                'position': get(player, 'games', 'position'),
                'number': safe_int(get(player, 'games', 'number')),
                'rating': safe_float(get(player, 'games', 'rating')),
                'captain': get(player, 'games', 'captain') or False,
                'substitute': get(player, 'games', 'substitute') or False,
                'starting': not (get(player, 'games', 'substitute') or False),

                'goals': safe_int(get(player, 'goals', 'total')) or 0,
                'assists': safe_int(get(player, 'goals', 'assists')) or 0,
                'goals_conceded': safe_int(get(player, 'goals', 'conceded')) or 0,
                'saves': safe_int(get(player, 'goals', 'saves')) or 0,

                'shots_total': safe_int(get(player, 'shots', 'total')) or 0,
                'shots_on': safe_int(get(player, 'shots', 'on')) or 0,

                'passes_total': safe_int(get(player, 'passes', 'total')),
                'passes_key': safe_int(get(player, 'passes', 'key')),
                'passes_accuracy': safe_int(get(player, 'passes', 'accuracy')),

                'tackles_total': safe_int(get(player, 'tackles', 'total')),
                'tackles_blocks': safe_int(get(player, 'tackles', 'blocks')),
                'tackles_interceptions': safe_int(get(player, 'tackles', 'interceptions')),

                'duels_total': safe_int(get(player, 'duels', 'total')) or 0,
                'duels_won': safe_int(get(player, 'duels', 'won')) or 0,

                'dribbles_attempts': safe_int(get(player, 'dribbles', 'attempts')) or 0,
                'dribbles_success': safe_int(get(player, 'dribbles', 'success')) or 0,
                'dribbles_past': safe_int(get(player, 'dribbles', 'past')) or 0,

                'fouls_drawn': safe_int(get(player, 'fouls', 'drawn')) or 0,
                'fouls_committed': safe_int(get(player, 'fouls', 'committed')) or 0,

                'yellow_cards': safe_int(get(player, 'cards', 'yellow')) or 0,
                'red_cards': safe_int(get(player, 'cards', 'red')) or 0,

                'penalty_won': safe_int(get(player, 'penalty', 'won')) or 0,
                'penalty_committed': safe_int(get(player, 'penalty', 'commited')) or 0,
                'penalty_scored': safe_int(get(player, 'penalty', 'scored')) or 0,
                'penalty_missed': safe_int(get(player, 'penalty', 'missed')) or 0,
                'penalty_saved': safe_int(get(player, 'penalty', 'saved')) or 0,

                'offsides': safe_int(player.get('offsides')) or 0
            }

        except Exception as e:
            logger.warning(f"Error extracting player stats: {e}")
            return None


class LineupExtractor(ILineupExtractor):
    """
    Lineup extractor with validation.
    Supports flat parquet data where each row is already a single lineup entry.
    """

    def __init__(self, validator: IDataValidator):
        self.validator = validator
        self.helper = NestedDataHelper()

    def extract(self, lineups_data: List[Dict[str, Any]], fixture_id: int) -> List[Dict[str, Any]]:
        """
        Extract lineups with validation.
        Data is now flat - each dict is already one player in a lineup.
        """
        extracted_lineups = []
        invalid_count = 0

        for lineup_entry in lineups_data:
            try:
                lineup_row = self._extract_single_lineup_flat(lineup_entry, fixture_id)

                if lineup_row is None:
                    continue

                if self.validator.validate(lineup_row):
                    extracted_lineups.append(lineup_row)
                else:
                    invalid_count += 1
                    logger.debug(
                        f"Lineup validation failed for {lineup_row.get('player_name')}: "
                        f"{self.validator.get_errors()}"
                    )

            except Exception as e:
                logger.warning(f"Error extracting lineup entry: {e}")
                continue

        if invalid_count > 0:
            logger.warning(
                f"Fixture {fixture_id}: {invalid_count} lineups failed validation"
            )

        return extracted_lineups

    def _extract_single_lineup_flat(self, entry: Dict[str, Any], fixture_id: int) -> Optional[Dict[str, Any]]:
        """Extract single lineup entry from flat parquet row."""
        try:
            get = self.helper.get_flat
            safe_int = self.helper.safe_int

            entry_type = entry.get('type', '')
            starting = entry_type == 'StartXI'

            return {
                'fixture_id': entry.get('fixture_id') or fixture_id,
                'team_id': safe_int(get(entry, 'team', 'id')),
                'team_name': entry.get('team_name'),
                'player_id': safe_int(entry.get('id') or get(entry, 'player', 'id')),
                'player_name': entry.get('name') or get(entry, 'player', 'name'),
                'position': entry.get('pos') or get(entry, 'player', 'pos'),
                'number': safe_int(entry.get('number') or get(entry, 'player', 'number')),
                'starting': starting,
            }
        except Exception as e:
            logger.warning(f"Error extracting single lineup: {e}")
            return None
