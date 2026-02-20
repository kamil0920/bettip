"""Window Ratio Feature Engineering — EMA_short / EMA_long for H2H Stats

GBDTs see absolute EMA levels but can't compute ratios internally — they approximate
with axis-aligned splits, losing precision. The DynamicsFeatureEngineer (S30) proved
this: _momentum_ratio (EMA_short/EMA_long) was selected 14/16 times for niche stats.

Gap: H2H stats (goals_scored, goals_conceded, points) only have
momentum = EMA_short - EMA_long (difference from MomentumFeatureEngineer).
This engineer adds the ratio EMA_short / EMA_long, which is inherently detrended,
scale-independent, and centered around 1.0.

Data: Loads match_stats.parquet (same source as DynamicsFeatureEngineer).
"""
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.data_collection.match_stats_utils import normalize_match_stats_columns
from src.features.engineers.base import BaseFeatureEngineer
from src.leagues import EUROPEAN_LEAGUES

logger = logging.getLogger(__name__)


class WindowRatioFeatureEngineer(BaseFeatureEngineer):
    """
    Generates EMA_short / EMA_long ratio features for H2H stats.

    Produces 18 features:
    - Per-side ratios (12): 6 stats x 2 sides (home/away)
    - Diff features (6): home ratio - away ratio per stat

    Stats: goals, goals_conceded, points, shots_on_target, possession, goals_per_shot

    All features use shift(1) to prevent data leakage.
    Ratios clipped to [0.3, 3.0] (same as DynamicsFeatureEngineer momentum_ratio).
    """

    STATS = ['goals', 'goals_conceded', 'points', 'shots_on_target', 'possession', 'goals_per_shot']
    SIDES = ['home', 'away']

    def __init__(
        self,
        short_ema: int = 3,
        long_ema: int = 12,
        min_matches: int = 3,
    ):
        self.short_ema = short_ema
        self.long_ema = long_ema
        self.min_matches = min_matches
        self.data_dir = Path("data/01-raw")

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create window ratio features from match stats."""
        matches = data.get('matches')
        if matches is None or matches.empty:
            return pd.DataFrame()

        match_stats = self._load_match_stats()
        if match_stats.empty:
            return pd.DataFrame()

        featured = self._build_features(match_stats)

        feature_cols = [
            c for c in featured.columns
            if c not in match_stats.columns or c == 'fixture_id'
        ]
        if 'fixture_id' not in feature_cols:
            feature_cols = ['fixture_id'] + feature_cols

        return featured[feature_cols]

    def _load_match_stats(self) -> pd.DataFrame:
        """Load match stats from all leagues."""
        all_stats = []
        for league in EUROPEAN_LEAGUES:
            league_dir = self.data_dir / league
            if not league_dir.exists():
                continue
            for season_dir in league_dir.iterdir():
                if not season_dir.is_dir():
                    continue
                stats_path = season_dir / 'match_stats.parquet'
                if stats_path.exists():
                    try:
                        df = pd.read_parquet(stats_path)
                        df = normalize_match_stats_columns(df)
                        df['league'] = league
                        all_stats.append(df)
                    except Exception as e:
                        logger.debug(f"Could not load {stats_path}: {e}")

        return pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()

    def _derive_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive points, goals_conceded, and goals_per_shot from raw match stats."""
        df = df.copy()

        for side, opp in [('home', 'away'), ('away', 'home')]:
            goals_col = f'{side}_goals'
            opp_goals_col = f'{opp}_goals'
            shots_col = f'{side}_shots'

            # Points: 3 for win, 1 for draw, 0 for loss
            if goals_col in df.columns and opp_goals_col in df.columns:
                df[f'{side}_points'] = np.where(
                    df[goals_col] > df[opp_goals_col], 3.0,
                    np.where(df[goals_col] == df[opp_goals_col], 1.0, 0.0)
                )
                df[f'{side}_goals_conceded'] = df[opp_goals_col]

            # Goals per shot: goals / shots (clinical finishing)
            if goals_col in df.columns and shots_col in df.columns:
                df[f'{side}_goals_per_shot'] = (
                    df[goals_col] / df[shots_col].replace(0, np.nan)
                )

        return df

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all window ratio features."""
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)
        df = self._derive_stats(df)

        short = self.short_ema
        long = self.long_ema
        min_p = self.min_matches

        # Column mapping: stat name -> actual column in df
        col_map = {
            'goals': '{side}_goals',
            'goals_conceded': '{side}_goals_conceded',
            'points': '{side}_points',
            'shots_on_target': '{side}_shots_on_target',
            'possession': '{side}_possession',
            'goals_per_shot': '{side}_goals_per_shot',
        }

        for stat in self.STATS:
            for side in self.SIDES:
                col = col_map[stat].format(side=side)
                team_col = f'{side}_team'
                if col not in df.columns or team_col not in df.columns:
                    continue

                ema_short = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).ewm(span=short, min_periods=min_p).mean()
                )
                ema_long = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).ewm(span=long, min_periods=min_p).mean()
                )

                df[f'{side}_{stat}_ratio'] = (
                    ema_short / ema_long.replace(0, np.nan)
                ).clip(0.3, 3.0)

            # Diff: home ratio - away ratio
            home_r = f'home_{stat}_ratio'
            away_r = f'away_{stat}_ratio'
            if home_r in df.columns and away_r in df.columns:
                df[f'{stat}_ratio_diff'] = df[home_r] - df[away_r]

        return df
