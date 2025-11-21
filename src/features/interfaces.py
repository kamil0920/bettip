"""Abstract interfaces for feature engineering components."""
from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class IDataLoader(ABC):
    """Interface for load data."""

    @abstractmethod
    def load(self, filepath: str) -> pd.DataFrame:
        """Load data from file."""
        pass


class IDataCleaner(ABC):
    """Interface for clean data."""

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data."""
        pass


class IFeatureEngineer(ABC):
    """Interface for create features."""

    @abstractmethod
    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features."""
        pass
