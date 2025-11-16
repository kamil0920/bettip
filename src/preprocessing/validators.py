"""Data validation implementations for preprocessing."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from src.preprocessing.interfaces import IDataValidator


class BaseValidator(IDataValidator, ABC):
    """
    Base validator class.
    """

    def __init__(self):
        self.errors: List[str] = []

    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate data and return True if valid.
        """
        self.errors.clear()
        return self._validate_impl(data)

    def get_errors(self) -> List[str]:
        """Return validation errors."""
        return self.errors.copy()

    @abstractmethod
    def _validate_impl(self, data: Dict[str, Any]) -> bool:
        """
        Implementation of validation logic.
        Must be implemented by subclasses.
        """
        pass


class FixtureValidator(BaseValidator):
    """Validator for fixture data."""

    REQUIRED_FIELDS = ['fixture_id', 'home_team_id', 'away_team_id']

    def _validate_impl(self, data: Dict[str, Any]) -> bool:
        """Validate fixture data - implements abstract method."""
        for field in self.REQUIRED_FIELDS:
            if field not in data or data[field] is None:
                self.errors.append(f"Missing required field: {field}")
                return False

        if not isinstance(data['fixture_id'], (int, str)):
            self.errors.append("fixture_id must be int or str")
            return False

        try:
            home_id = int(data['home_team_id'])
            away_id = int(data['away_team_id'])

            if home_id <= 0 or away_id <= 0:
                self.errors.append("Team IDs must be positive")
                return False

            if home_id == away_id:
                self.errors.append("Home and away team IDs cannot be the same")
                return False

        except (ValueError, TypeError):
            self.errors.append("Team IDs must be numeric")
            return False

        return True


class PlayerStatsValidator(BaseValidator):
    """Validator for player statistics."""

    REQUIRED_FIELDS = ['fixture_id', 'player_id', 'team_id']
    NUMERIC_FIELDS = ['minutes', 'goals', 'assists', 'yellow_cards', 'red_cards']
    OPTIONAL_NUMERIC_FIELDS = [
        'shots_total', 'shots_on', 'passes_total', 'passes_key',
        'tackles_total', 'duels_total', 'dribbles_attempts',
        'fouls_drawn', 'fouls_committed', 'rating'
    ]

    def _validate_impl(self, data: Dict[str, Any]) -> bool:
        """Validate player stats."""
        for field in self.REQUIRED_FIELDS:
            if field not in data or data[field] is None:
                self.errors.append(f"Missing required field: {field}")
                return False

        try:
            fixture_id = int(data['fixture_id'])
            player_id = int(data['player_id'])
            team_id = int(data['team_id'])

            if fixture_id <= 0 or player_id <= 0 or team_id <= 0:
                self.errors.append("IDs must be positive")
                return False

        except (ValueError, TypeError):
            self.errors.append("IDs must be numeric")
            return False

        for field in self.NUMERIC_FIELDS:
            if field in data and data[field] is not None:
                if not self._validate_numeric_field(field, data[field], min_value=0):
                    return False

        for field in self.OPTIONAL_NUMERIC_FIELDS:
            if field in data and data[field] is not None:
                if not self._validate_numeric_field(field, data[field], min_value=0, allow_float=True):
                    return False

        if 'minutes' in data and data['minutes'] is not None:
            try:
                minutes = float(data['minutes'])
                if minutes < 0 or minutes > 120:
                    self.errors.append(f"Minutes must be between 0 and 120, got {minutes}")
                    return False
            except (ValueError, TypeError):
                pass

        if 'rating' in data and data['rating'] is not None:
            try:
                rating = float(data['rating'])
                if rating < 0 or rating > 10:
                    self.errors.append(f"Rating must be between 0 and 10, got {rating}")
                    return False
            except (ValueError, TypeError):
                pass

        return True

    def _validate_numeric_field(self, field: str, value: Any, min_value: float = None, max_value: float = None, allow_float: bool = False) -> bool:
        """Helper to validate numeric fields."""
        try:
            num_value = float(value)

            if not allow_float and not num_value.is_integer():
                self.errors.append(f"{field} must be an integer")
                return False

            if min_value is not None and num_value < min_value:
                self.errors.append(f"{field} cannot be less than {min_value}, got {num_value}")
                return False

            if max_value is not None and num_value > max_value:
                self.errors.append(f"{field} cannot be greater than {max_value}, got {num_value}")
                return False

            return True

        except (ValueError, TypeError):
            self.errors.append(f"{field} must be numeric, got {type(value).__name__}")
            return False


class EventValidator(BaseValidator):
    """Validator for event data."""

    REQUIRED_FIELDS = ['fixture_id', 'type', 'team_id']
    VALID_EVENT_TYPES = [
        'Goal', 'Card', 'subst', 'Var', 'Penalty',
        'Corner', 'Offside', 'Foul', 'Shot'
    ]

    def _validate_impl(self, data: Dict[str, Any]) -> bool:
        """Validate event data."""
        for field in self.REQUIRED_FIELDS:
            if field not in data or data[field] is None:
                self.errors.append(f"Missing required field: {field}")
                return False

        event_type = data.get('type')
        if event_type not in self.VALID_EVENT_TYPES:
            self.errors.append(f"Invalid event type: {event_type}")
            return False

        if 'time_elapsed' in data and data['time_elapsed'] is not None:
            try:
                time_elapsed = int(data['time_elapsed'])
                if time_elapsed < 0 or time_elapsed > 150:
                    self.errors.append(f"time_elapsed must be between 0 and 150")
                    return False
            except (ValueError, TypeError):
                self.errors.append("time_elapsed must be numeric")
                return False

        return True


class LineupValidator(BaseValidator):
    """Validator for lineup data."""

    REQUIRED_FIELDS = ['fixture_id', 'team_id', 'player_id']

    def _validate_impl(self, data: Dict[str, Any]) -> bool:
        """Validate lineup data - implements abstract method."""
        for field in self.REQUIRED_FIELDS:
            if field not in data or data[field] is None:
                self.errors.append(f"Missing required field: {field}")
                return False

        try:
            fixture_id = int(data['fixture_id'])
            team_id = int(data['team_id'])
            player_id = int(data['player_id'])

            if fixture_id <= 0 or team_id <= 0 or player_id <= 0:
                self.errors.append("IDs must be positive")
                return False

        except (ValueError, TypeError):
            self.errors.append("IDs must be numeric")
            return False

        if 'starting' in data:
            if not isinstance(data['starting'], bool):
                self.errors.append("starting must be boolean")
                return False

        return True
