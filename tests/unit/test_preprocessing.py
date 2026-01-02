"""Unit tests for preprocessing module."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.preprocessing.validators import (
    FixtureValidator,
    EventValidator,
    PlayerStatsValidator,
    LineupValidator
)
from src.preprocessing.extractors import NestedDataHelper, FixtureExtractor
from src.preprocessing.loaders import JSONDataLoader, _extract_fixture_id
from src.preprocessing.exceptions import FileLoadError


class TestNestedDataHelper:
    """Tests for NestedDataHelper utility class."""

    def test_get_nested_simple(self):
        """Test getting nested value from dict."""
        data = {'a': {'b': {'c': 'value'}}}
        result = NestedDataHelper.get_nested(data, 'a', 'b', 'c')
        assert result == 'value'

    def test_get_nested_missing_key(self):
        """Test getting missing nested key returns None."""
        data = {'a': {'b': 'value'}}
        result = NestedDataHelper.get_nested(data, 'a', 'c')
        assert result is None

    def test_get_nested_none_intermediate(self):
        """Test handling None in intermediate path."""
        data = {'a': None}
        result = NestedDataHelper.get_nested(data, 'a', 'b')
        assert result is None

    def test_safe_int_valid(self):
        """Test safe int conversion with valid values."""
        assert NestedDataHelper.safe_int(10) == 10
        assert NestedDataHelper.safe_int("20") == 20
        assert NestedDataHelper.safe_int(3.7) == 3

    def test_safe_int_none(self):
        """Test safe int conversion with None."""
        assert NestedDataHelper.safe_int(None) is None

    def test_safe_int_invalid(self):
        """Test safe int conversion with invalid value."""
        assert NestedDataHelper.safe_int("abc") is None

    def test_safe_float_valid(self):
        """Test safe float conversion with valid values."""
        assert NestedDataHelper.safe_float(10.5) == 10.5
        assert NestedDataHelper.safe_float("7.3") == 7.3
        assert NestedDataHelper.safe_float(3) == 3.0

    def test_safe_float_none(self):
        """Test safe float conversion with None."""
        assert NestedDataHelper.safe_float(None) is None


class TestFixtureValidator:
    """Tests for FixtureValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = FixtureValidator()

    def test_valid_fixture(self):
        """Test validation of valid fixture data."""
        data = {
            'fixture_id': 12345,
            'home_team_id': 100,
            'away_team_id': 200,
            'date': '2024-01-01',
            'ft_home': 2,
            'ft_away': 1
        }
        assert self.validator.validate(data) is True
        assert len(self.validator.get_errors()) == 0

    def test_missing_fixture_id(self):
        """Test validation fails when fixture_id is missing."""
        data = {
            'home_team_id': 100,
            'away_team_id': 200
        }
        assert self.validator.validate(data) is False
        assert 'fixture_id' in str(self.validator.get_errors())

    def test_missing_home_team_id(self):
        """Test validation fails when home_team_id is missing."""
        data = {
            'fixture_id': 12345,
            'away_team_id': 200
        }
        assert self.validator.validate(data) is False
        assert 'home_team_id' in str(self.validator.get_errors())

    def test_same_team_ids(self):
        """Test validation fails when home and away team IDs are same."""
        data = {
            'fixture_id': 12345,
            'home_team_id': 100,
            'away_team_id': 100
        }
        assert self.validator.validate(data) is False
        assert 'same' in str(self.validator.get_errors()).lower()

    def test_negative_team_id(self):
        """Test validation fails with negative team ID."""
        data = {
            'fixture_id': 12345,
            'home_team_id': -1,
            'away_team_id': 200
        }
        assert self.validator.validate(data) is False


class TestEventValidator:
    """Tests for EventValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = EventValidator()

    def test_valid_goal_event(self):
        """Test validation of valid goal event."""
        data = {
            'fixture_id': 12345,
            'type': 'Goal',
            'team_id': 100,
            'time_elapsed': 45
        }
        assert self.validator.validate(data) is True

    def test_valid_card_event(self):
        """Test validation of valid card event."""
        data = {
            'fixture_id': 12345,
            'type': 'Card',
            'team_id': 100,
            'detail': 'Yellow Card'
        }
        assert self.validator.validate(data) is True

    def test_invalid_event_type(self):
        """Test validation fails for invalid event type."""
        data = {
            'fixture_id': 12345,
            'type': 'InvalidType',
            'team_id': 100
        }
        assert self.validator.validate(data) is False
        assert 'Invalid event type' in str(self.validator.get_errors())

    def test_time_elapsed_out_of_range(self):
        """Test validation fails for time_elapsed > 150."""
        data = {
            'fixture_id': 12345,
            'type': 'Goal',
            'team_id': 100,
            'time_elapsed': 200
        }
        assert self.validator.validate(data) is False


class TestPlayerStatsValidator:
    """Tests for PlayerStatsValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PlayerStatsValidator()

    def test_valid_player_stats(self):
        """Test validation of valid player stats."""
        data = {
            'fixture_id': 12345,
            'player_id': 1001,
            'team_id': 100,
            'minutes': 90,
            'goals': 1,
            'assists': 0,
            'yellow_cards': 1,
            'red_cards': 0,
            'rating': 7.5
        }
        assert self.validator.validate(data) is True

    def test_invalid_minutes(self):
        """Test validation fails for minutes > 150."""
        data = {
            'fixture_id': 12345,
            'player_id': 1001,
            'team_id': 100,
            'minutes': 200
        }
        assert self.validator.validate(data) is False

    def test_invalid_rating(self):
        """Test validation fails for rating > 10."""
        data = {
            'fixture_id': 12345,
            'player_id': 1001,
            'team_id': 100,
            'rating': 15.0
        }
        assert self.validator.validate(data) is False


class TestLineupValidator:
    """Tests for LineupValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = LineupValidator()

    def test_valid_lineup(self):
        """Test validation of valid lineup data."""
        data = {
            'fixture_id': 12345,
            'team_id': 100,
            'player_id': 1001,
            'starting': True
        }
        assert self.validator.validate(data) is True

    def test_invalid_starting_type(self):
        """Test validation fails when starting is not boolean."""
        data = {
            'fixture_id': 12345,
            'team_id': 100,
            'player_id': 1001,
            'starting': 'yes'  # Should be boolean
        }
        assert self.validator.validate(data) is False


class TestJSONDataLoader:
    """Tests for JSONDataLoader."""

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises FileLoadError."""
        loader = JSONDataLoader()
        with pytest.raises(FileLoadError):
            loader.load(Path("/nonexistent/path.json"))

    @patch('builtins.open', side_effect=Exception("IO Error"))
    def test_load_io_error(self, mock_open):
        """Test handling IO errors."""
        loader = JSONDataLoader()
        with pytest.raises(FileLoadError):
            loader.load(Path("/some/path.json"))


class TestFixtureExtractor:
    """Tests for FixtureExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = FixtureValidator()
        self.extractor = FixtureExtractor(self.validator)

    def test_extract_valid_fixture(self):
        """Test extraction of valid fixture data."""
        fixture_data = {
            'fixture': {
                'id': 12345,
                'date': '2024-01-01T15:00:00+00:00',
                'timestamp': 1704114000,
                'referee': 'John Doe',
                'venue': {'id': 1, 'name': 'Stadium A'},
                'status': {'short': 'FT'}
            },
            'league': {
                'id': 39,
                'round': 'Regular Season - 1'
            },
            'teams': {
                'home': {'id': 100, 'name': 'Team A'},
                'away': {'id': 200, 'name': 'Team B'}
            },
            'score': {
                'fulltime': {'home': 2, 'away': 1},
                'halftime': {'home': 1, 'away': 0}
            }
        }

        result = self.extractor.extract(fixture_data)

        assert result is not None
        assert result['fixture_id'] == 12345
        assert result['home_team_id'] == 100
        assert result['away_team_id'] == 200
        assert result['ft_home'] == 2
        assert result['ft_away'] == 1
        assert result['home_team_name'] == 'Team A'

    def test_extract_invalid_fixture(self):
        """Test extraction returns None for invalid data."""
        fixture_data = {
            'fixture': {'id': None},  # Invalid - no ID
            'teams': {
                'home': {'id': 100},
                'away': {'id': 100}  # Same as home - invalid
            }
        }

        result = self.extractor.extract(fixture_data)
        assert result is None


class TestExtractFixtureId:
    """Tests for _extract_fixture_id helper function."""

    def test_extract_from_dict(self):
        """Test extracting fixture ID from dict."""
        value = {'id': 12345, 'date': '2024-01-01'}
        assert _extract_fixture_id(value) == 12345

    def test_extract_from_string_dict(self):
        """Test extracting fixture ID from string representation of dict."""
        value = "{'id': 12345, 'date': '2024-01-01'}"
        assert _extract_fixture_id(value) == 12345

    def test_extract_from_none(self):
        """Test extracting fixture ID from None returns None."""
        assert _extract_fixture_id(None) is None

    def test_extract_from_dict_without_id(self):
        """Test extracting from dict without 'id' key returns None."""
        value = {'date': '2024-01-01', 'home_team': 'Liverpool'}
        assert _extract_fixture_id(value) is None

    def test_extract_from_invalid_string(self):
        """Test extracting from invalid string returns None."""
        value = "not a dict"
        assert _extract_fixture_id(value) is None

    def test_extract_from_complex_nested_string(self):
        """Test extracting from complex string with nested dicts."""
        value = "{'id': 1378969, 'date': '2025-08-15T19:00:00+00:00', 'home_team': 'Liverpool', 'score': {'home': 4, 'away': 2}}"
        assert _extract_fixture_id(value) == 1378969
