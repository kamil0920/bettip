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
    """Helper for working with nested structures."""

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
        """Implementation of extraction logic."""
        get = self.helper.get_nested

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
    """

    def __init__(self, validator: IDataValidator):
        self.validator = validator
        self.helper = NestedDataHelper()

    def extract(self, events_data: List[Dict[str, Any]], fixture_id: int) -> List[Dict[str, Any]]:
        """
        Extract events with validation.
        """
        extracted_events = []
        invalid_count = 0

        for idx, event in enumerate(events_data):
            try:
                if event.get('type') in ['Goal', 'Card']:
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
        """Extract single event data."""
        return {
            'fixture_id': fixture_id,
            'type': event.get('type'),
            'detail': event.get('detail'),
            'time_elapsed': self.helper.get_nested(event, 'time', 'elapsed'),
            'team_id': self.helper.get_nested(event, 'team', 'id'),
            'player_id': self.helper.get_nested(event, 'player', 'id'),
            'player_name': self.helper.get_nested(event, 'player', 'name'),
            'assist_id': self.helper.get_nested(event, 'assist', 'id'),
            'assist_name': self.helper.get_nested(event, 'assist', 'name'),
        }


class PlayerStatsExtractor(IPlayerStatsExtractor):
    """
    Player statistics extractor implementation.
    """

    def __init__(self, validator: IDataValidator):
        self.validator = validator
        self.helper = NestedDataHelper()

    def extract(self, players_data: List[Dict[str, Any]], fixture_id: int) -> List[Dict[str, Any]]:
        """
        Extract player statistics.
        """
        extracted_stats = []

        for player in players_data:
            team_id = player['team']['id']
            team_name = player['team']['name']

            for player_entry in player['players']:
                player_stat = self._extract_single_player(
                    player_entry, fixture_id, team_id, team_name
                )

                if player_stat and self.validator.validate(player_stat):
                    extracted_stats.append(player_stat)
                else:
                    if player_stat:
                        logger.debug(f"Player stat validation failed: {self.validator.get_errors()}")

        return extracted_stats

    def _extract_single_player(self, player_entry: Dict[str, Any], fixture_id: int, team_id: int, team_name: str) -> Optional[Dict[str, Any]]:
        """Extract single player data."""
        try:
            player = player_entry['player']
            stats = player_entry['statistics'][0] if player_entry['statistics'] else {}

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

            return {
                'fixture_id': fixture_id,
                'player_id': player['id'],
                'player_name': player['name'],
                'team_id': team_id,
                'team_name': team_name,

                'minutes': games.get('minutes') or 0,
                'position': games.get('position'),
                'number': games.get('number'),
                'rating': self.helper.safe_float(games.get('rating')),
                'captain': games.get('captain', False),
                'substitute': games.get('substitute', False),
                'starting': not games.get('substitute', False),

                'goals': goals_data.get('total') or 0,
                'assists': goals_data.get('assists') or 0,
                'goals_conceded': goals_data.get('conceded') or 0,
                'saves': goals_data.get('saves') or 0,

                'shots_total': shots.get('total') or 0,
                'shots_on': shots.get('on') or 0,

                'passes_total': self.helper.safe_int(passes.get('total')),
                'passes_key': self.helper.safe_int(passes.get('key')),
                'passes_accuracy': self.helper.safe_int(passes.get('accuracy')),

                'tackles_total': self.helper.safe_int(tackles.get('total')),
                'tackles_blocks': self.helper.safe_int(tackles.get('blocks')),
                'tackles_interceptions': self.helper.safe_int(tackles.get('interceptions')),

                'duels_total': duels.get('total') or 0,
                'duels_won': duels.get('won') or 0,

                'dribbles_attempts': dribbles.get('attempts') or 0,
                'dribbles_success': dribbles.get('success') or 0,
                'dribbles_past': dribbles.get('past') or 0,

                'fouls_drawn': fouls.get('drawn') or 0,
                'fouls_committed': fouls.get('committed') or 0,

                'yellow_cards': cards.get('yellow', 0),
                'red_cards': cards.get('red', 0),

                'penalty_won': penalty.get('won') or 0,
                'penalty_committed': penalty.get('commited') or 0,
                'penalty_scored': penalty.get('scored', 0),
                'penalty_missed': penalty.get('missed', 0),
                'penalty_saved': penalty.get('saved', 0),

                'offsides': stats.get('offsides') or 0
            }

        except Exception as e:
            logger.warning(f"Error extracting player stats: {e}")
            return None


class LineupExtractor(ILineupExtractor):
    """
    Lineup extractor with validation.
    """

    def __init__(self, validator: IDataValidator):
        self.validator = validator
        self.helper = NestedDataHelper()

    def extract(self, lineups_data: List[Dict[str, Any]], fixture_id: int) -> List[Dict[str, Any]]:
        """
        Extract lineups with validation.
        """
        extracted_lineups = []
        invalid_count = 0

        for team_lineup in lineups_data:
            try:
                team_id = team_lineup['team']['id']
                team_name = team_lineup['team']['name']
                formation = team_lineup.get('formation', '')

                formation_row = {'formation': formation}

                extracted_lineups.append(formation_row)

                for player_data in team_lineup.get('startXI', []):
                    lineup_row = self._extract_single_lineup(
                        player_data, fixture_id, team_id, team_name, starting=True
                    )

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

                for player_data in team_lineup.get('substitutes', []):
                    lineup_row = self._extract_single_lineup(
                        player_data, fixture_id, team_id, team_name, starting=False
                    )

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
                logger.warning(f"Error extracting lineup for team: {e}")
                continue

        if invalid_count > 0:
            logger.warning(
                f"Fixture {fixture_id}: {invalid_count} lineups failed validation"
            )

        return extracted_lineups

    def _extract_single_lineup(self, player_data: Dict[str, Any], fixture_id: int, team_id: int, team_name: str, starting: bool) -> Optional[Dict[str, Any]]:
        """Extract single lineup entry."""
        try:
            player = player_data['player']

            return {
                'fixture_id': fixture_id,
                'team_id': team_id,
                'team_name': team_name,
                'player_id': player['id'],
                'player_name': player['name'],
                'position': player.get('pos'),
                'number': player.get('number'),
                'starting': starting,
            }
        except Exception as e:
            logger.warning(f"Error extracting single lineup: {e}")
            return None
