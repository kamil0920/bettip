"""
STL Decomposition Feature Engineering — Trend Strength per Team

GBDTs see point-in-time levels and distributional features but cannot separate
a time series into trend, seasonal, and residual components. STL (Seasonal and
Trend decomposition using Loess) isolates each component, and *trend strength*
(1 − Var(residual) / Var(trend + residual)) quantifies how much of the signal is
driven by a persistent trend vs noise.

- High trend strength → team is on a consistent trajectory (improving or declining).
- Low trend strength → team's stats fluctuate around a flat level.

This is complementary to dynamics (distributional shape), entropy (ordinal complexity),
and spectral (frequency-domain periodicity) engineers.

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


def _trend_strength(x: np.ndarray, period: int = 10) -> float:
    """Trend strength via STL decomposition: 1 − Var(residual) / Var(trend + residual).

    Returns value in [0, 1]. Higher = stronger trend, lower = noise-dominated.
    Returns NaN if the series is too short or STL fails.
    """
    from statsmodels.tsa.seasonal import STL

    x = x[~np.isnan(x)]
    if len(x) < max(period * 2, 20):
        return np.nan
    if np.std(x) < 1e-10:
        return 0.0
    try:
        stl = STL(x, period=period, robust=True)
        result = stl.fit()
        var_resid = np.var(result.resid)
        var_detrended = np.var(result.trend + result.resid)
        if var_detrended < 1e-10:
            return 0.0
        return max(0.0, 1.0 - var_resid / var_detrended)
    except Exception:
        return np.nan


class DecompositionFeatureEngineer(BaseFeatureEngineer):
    """STL decomposition features: trend strength per team per stat.

    Produces ~20 features:
    - Trend strength per side per stat (10) + diffs (5) + sums (5)

    Stats: fouls, shots, corners, cards, goals (5 stats x 2 sides = 10 base series)

    All features use shift(1) to prevent data leakage.
    """

    STATS = ['fouls', 'shots', 'corners', 'cards', 'goals']
    SIDES = ['home', 'away']

    def __init__(
        self,
        period: int = 10,
        min_obs: int = 20,
    ):
        self.period = period
        self.min_obs = min_obs
        self.data_dir = Path("data/01-raw")

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create STL trend strength features from match stats."""
        matches = data.get('matches')
        if matches is None or matches.empty:
            return pd.DataFrame()

        match_stats = self._load_match_stats()
        if match_stats.empty:
            return pd.DataFrame()

        match_stats = self._derive_cards(match_stats)
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

    def _derive_cards(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive home_cards/away_cards from yellow + red if not present."""
        for side in self.SIDES:
            col = f'{side}_cards'
            if col not in df.columns:
                yellow = f'{side}_yellow_cards'
                red = f'{side}_red_cards'
                if yellow in df.columns and red in df.columns:
                    df[col] = df[yellow].fillna(0) + df[red].fillna(0)
        return df

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build trend strength features via STL decomposition."""
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)

        period = self.period
        min_obs = self.min_obs

        for stat in self.STATS:
            for side in self.SIDES:
                col = f'{side}_{stat}'
                if col not in df.columns:
                    continue
                team_col = f'{side}_team'
                if team_col not in df.columns:
                    continue

                # Trend strength: shift(1) prevents look-ahead bias.
                # Rolling window = min_obs ensures enough data for STL.
                ts_col = f'{side}_{stat}_trend_strength'
                df[ts_col] = (
                    df.groupby(team_col)[col]
                    .transform(
                        lambda x: x.shift(1)
                        .rolling(min_obs, min_periods=min_obs)
                        .apply(lambda v: _trend_strength(v, period), raw=True)
                    )
                ).clip(0, 1).fillna(0.5)

        # Cross-side features: diff and mean for trend strength
        for stat in self.STATS:
            h_ts = f'home_{stat}_trend_strength'
            a_ts = f'away_{stat}_trend_strength'
            if h_ts in df.columns and a_ts in df.columns:
                df[f'{stat}_trend_strength_diff'] = df[h_ts] - df[a_ts]
                df[f'{stat}_trend_strength_avg'] = (df[h_ts] + df[a_ts]) / 2

        n_features = len([c for c in df.columns if 'trend_strength' in c])
        logger.info(f"DecompositionFeatureEngineer: created {n_features} features")
        return df
