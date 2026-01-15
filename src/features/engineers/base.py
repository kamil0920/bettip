"""Base class for feature engineers."""
from typing import Dict

import pandas as pd

from src.features.interfaces import IFeatureEngineer


class BaseFeatureEngineer(IFeatureEngineer):
    """Base class for feature engineers."""

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Template method pattern."""
        raise NotImplementedError
