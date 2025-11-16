"""Abstract interfaces for data preprocessing components."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class IDataLoader(ABC):
    """Interface for data loading."""

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Load data from file."""
        pass


class IDataParser(ABC):
    """Interface for data parsing."""

    @abstractmethod
    def parse(self, raw_data: Any) -> Optional[Dict[str, Any]]:
        """Parse raw data."""
        pass


class IDataValidator(ABC):
    """Interface for data validation."""

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data."""
        pass

    @abstractmethod
    def get_errors(self) -> List[str]:
        """Return validation errors."""
        pass


class IDataWriter(ABC):
    """Interface for writing data."""

    @abstractmethod
    def write(self, data: pd.DataFrame, path: Path) -> None:
        """Save data to file."""
        pass


class IFixtureExtractor(ABC):
    """Interface for fixture extraction."""

    @abstractmethod
    def extract(self, fixture_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract single fixture data.

        Args:
            fixture_data: Raw fixture data from API

        Returns:
            Extracted fixture dictionary or None if extraction fails
        """
        pass


class IEventExtractor(ABC):
    """Interface for event extraction."""

    @abstractmethod
    def extract(self, events_data: List[Dict[str, Any]], fixture_id: int) -> List[Dict[str, Any]]:
        """
        Extract events for a fixture.

        Args:
            events_data: List of raw event data
            fixture_id: Fixture identifier

        Returns:
            List of extracted event dictionaries
        """
        pass


class IPlayerStatsExtractor(ABC):
    """Interface for player statistics extraction."""

    @abstractmethod
    def extract(self, players_data: List[Dict[str, Any]], fixture_id: int) -> List[Dict[str, Any]]:
        """
        Extract player statistics for a fixture.

        Args:
            players_data: List of team data with players
            fixture_id: Fixture identifier

        Returns:
            List of extracted player stat dictionaries
        """
        pass


class ILineupExtractor(ABC):
    """Interface for lineup extraction."""

    @abstractmethod
    def extract(self, lineups_data: List[Dict[str, Any]], fixture_id: int) -> List[Dict[str, Any]]:
        """
        Extract lineups for a fixture.

        Args:
            lineups_data: List of team lineup data
            fixture_id: Fixture identifier

        Returns:
            List of extracted lineup dictionaries
        """
        pass
