"""Base class for feature engineers."""

import logging as _logging
from pathlib import Path as _Path
from typing import Dict

import pandas as pd

from src.data_collection.match_stats_utils import normalize_match_stats_columns
from src.features.interfaces import IFeatureEngineer
from src.leagues import ALL_LEAGUES

_logger = _logging.getLogger(__name__)


class BaseFeatureEngineer(IFeatureEngineer):
    """Base class for feature engineers."""

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Template method pattern."""
        raise NotImplementedError


class MatchStatsLoaderMixin:
    """Mixin providing shared match_stats loading from raw parquet files.

    Requires ``self.data_dir`` (Path) to be set on the class.
    """

    data_dir: _Path

    def _load_match_stats(self) -> pd.DataFrame:
        """Load match stats from all leagues."""
        all_stats = []
        for league in ALL_LEAGUES:
            league_dir = self.data_dir / league
            if not league_dir.exists():
                continue
            for season_dir in league_dir.iterdir():
                if not season_dir.is_dir():
                    continue
                stats_path = season_dir / "match_stats.parquet"
                if stats_path.exists():
                    try:
                        df = pd.read_parquet(stats_path)
                        df = normalize_match_stats_columns(df)
                        df["league"] = league
                        all_stats.append(df)
                    except Exception as e:
                        _logger.debug(f"Could not load {stats_path}: {e}")

        return pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
