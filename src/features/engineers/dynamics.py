"""
Dynamics Feature Engineering — Higher-Order Distributional Features

GBDTs see levels (EMAs) but not dynamics (trends, volatility shape, regime changes).
This engineer adds three signal dimensions beyond what niche_markets.py and niche_derived.py provide:

1. **Distributional features** (skewness, kurtosis, CoV): Capture asymmetry, tail weight,
   and normalized dispersion. Two teams with identical mean fouls but different skewness
   have very different OVER/UNDER distributions.

2. **Niche stat momentum** (EMA_short/EMA_long ratio, first differences): Captures trend
   direction for fouls/shots/corners/cards — extending MomentumFeatureEngineer (goals only).

3. **Regime detection** (variance ratio = recent_std / long_std): Values >1 signal
   destabilizing regime changes, <1 signal stabilization. Critical for line markets
   where bookmakers price the mean but not the variance dynamics.

Data: Loads match_stats.parquet (same source as NicheStatDerivedFeatureEngineer).
"""
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_collection.match_stats_utils import normalize_match_stats_columns
from src.features.engineers.base import BaseFeatureEngineer
from src.leagues import EUROPEAN_LEAGUES

logger = logging.getLogger(__name__)


class DynamicsFeatureEngineer(BaseFeatureEngineer):
    """
    Generates higher-order distributional, momentum, regime, ratio, Hurst,
    and damped-trend features for niche markets.

    Produces ~84 features across 6 categories:
    - Distributional (24): rolling skewness, kurtosis, CoV per side per stat
    - Momentum (20): EMA ratio, first diff per side per stat, plus diff features
    - Regime (12): variance ratio per side per stat, plus diff features
    - Match ratio (4): home/away EMA ratio per stat (scale-invariant imbalance)
    - Hurst exponent (12): rolling R/S Hurst per side per stat, plus diff features
    - Damped trend (12): Holt's damped trend per side per stat, plus diff features

    All features use shift(1) to prevent data leakage.
    """

    NICHE_STATS = ['fouls', 'shots', 'corners', 'cards']
    SIDES = ['home', 'away']

    def __init__(
        self,
        window: int = 10,
        short_ema: int = 5,
        long_ema: int = 15,
        long_window: int = 20,
        damping_factor: float = 0.9,
        hurst_window: int = 15,
        min_matches: int = 3,
    ):
        self.window = window
        self.short_ema = short_ema
        self.long_ema = long_ema
        self.long_window = long_window
        self.damping_factor = damping_factor
        self.hurst_window = hurst_window
        self.min_matches = min_matches
        self.data_dir = Path("data/01-raw")

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create dynamics features from match stats."""
        matches = data.get('matches')
        if matches is None or matches.empty:
            return pd.DataFrame()

        match_stats = self._load_match_stats()
        if match_stats.empty:
            return pd.DataFrame()

        match_stats = self._derive_cards(match_stats)
        featured = self._build_features(match_stats)

        # Return only new feature columns + fixture_id
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
                if yellow in df.columns:
                    red = df.get(f'{side}_red_cards', 0)
                    df[col] = df[yellow].fillna(0) + pd.Series(red).fillna(0)
        return df

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all dynamics features."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)

        # Ensure cards are derived before feature building
        df = self._derive_cards(df)

        self._build_distributional_features(df)
        self._build_momentum_features(df)
        self._build_regime_features(df)
        self._build_match_ratio_features(df)
        self._build_hurst_features(df)
        self._build_damped_trend_features(df)

        return df

    # --- Category 1: Distributional Features (24) ---

    def _build_distributional_features(self, df: pd.DataFrame) -> None:
        """Build rolling skewness, kurtosis, and CoV features."""
        window = self.window
        min_p = self.min_matches

        for stat in self.NICHE_STATS:
            for side in self.SIDES:
                col = f'{side}_{stat}'
                team_col = f'{side}_team'
                if col not in df.columns or team_col not in df.columns:
                    continue

                # Skewness: asymmetry of the distribution
                df[f'{side}_{stat}_skewness'] = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=min_p).skew()
                ).clip(-3, 3)

                # Kurtosis: tail weight (excess kurtosis, normal=0)
                df[f'{side}_{stat}_kurtosis'] = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=min_p).kurt()
                ).clip(-3, 10)

                # CoV: std/mean — scale-independent dispersion
                rolled_mean = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=min_p).mean()
                )
                rolled_std = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=min_p).std()
                )
                df[f'{side}_{stat}_cov'] = (
                    rolled_std / rolled_mean.replace(0, np.nan)
                ).clip(0, 5)

    # --- Category 2: Niche Stat Momentum (20) ---

    def _build_momentum_features(self, df: pd.DataFrame) -> None:
        """Build EMA ratio and first-difference momentum features."""
        short = self.short_ema
        long = self.long_ema
        min_p = self.min_matches

        for stat in self.NICHE_STATS:
            for side in self.SIDES:
                col = f'{side}_{stat}'
                team_col = f'{side}_team'
                if col not in df.columns or team_col not in df.columns:
                    continue

                # EMA short and long (shifted for leakage prevention)
                ema_short = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).ewm(span=short, min_periods=min_p).mean()
                )
                ema_long = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).ewm(span=long, min_periods=min_p).mean()
                )

                # Momentum ratio: EMA_short / EMA_long — inherently detrended
                df[f'{side}_{stat}_momentum_ratio'] = (
                    ema_short / ema_long.replace(0, np.nan)
                ).clip(0.3, 3.0)

                # First difference of EMA: rate of change (acceleration)
                df[f'{side}_{stat}_first_diff'] = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).ewm(span=short, min_periods=min_p).mean().diff()
                )

            # Momentum ratio diff: home - away
            home_mr = f'home_{stat}_momentum_ratio'
            away_mr = f'away_{stat}_momentum_ratio'
            if home_mr in df.columns and away_mr in df.columns:
                df[f'{stat}_momentum_ratio_diff'] = df[home_mr] - df[away_mr]

    # --- Category 3: Regime Detection (12) ---

    def _build_regime_features(self, df: pd.DataFrame) -> None:
        """Build variance ratio features for regime change detection."""
        short_w = self.window
        long_w = self.long_window
        min_p = self.min_matches

        for stat in self.NICHE_STATS:
            for side in self.SIDES:
                col = f'{side}_{stat}'
                team_col = f'{side}_team'
                if col not in df.columns or team_col not in df.columns:
                    continue

                recent_std = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=short_w, min_periods=min_p).std()
                )
                long_std = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=long_w, min_periods=min_p).std()
                )

                # Variance ratio: >1 = destabilizing, <1 = stabilizing
                df[f'{side}_{stat}_variance_ratio'] = (
                    recent_std / long_std.replace(0, np.nan)
                ).clip(0.1, 5.0)

            # Variance ratio diff: home - away
            home_vr = f'home_{stat}_variance_ratio'
            away_vr = f'away_{stat}_variance_ratio'
            if home_vr in df.columns and away_vr in df.columns:
                df[f'{stat}_variance_ratio_diff'] = df[home_vr] - df[away_vr]

    # --- Category 4: Match Ratio Features (4) ---

    def _build_match_ratio_features(self, df: pd.DataFrame) -> None:
        """Build home/away EMA ratio — scale-invariant match imbalance signal.

        Ratios detrend across leagues (Turkish Super Lig 28 fouls/game vs
        Bundesliga 20), unlike additive diffs which are league-dependent.
        """
        short = self.short_ema
        min_p = self.min_matches

        for stat in self.NICHE_STATS:
            home_col = f'home_{stat}'
            away_col = f'away_{stat}'
            home_team_col = 'home_team'
            away_team_col = 'away_team'
            if (home_col not in df.columns or away_col not in df.columns
                    or home_team_col not in df.columns or away_team_col not in df.columns):
                continue

            home_ema = df.groupby(home_team_col)[home_col].transform(
                lambda x: x.shift(1).ewm(span=short, min_periods=min_p).mean()
            )
            away_ema = df.groupby(away_team_col)[away_col].transform(
                lambda x: x.shift(1).ewm(span=short, min_periods=min_p).mean()
            )
            df[f'{stat}_match_ratio'] = (
                home_ema / away_ema.replace(0, np.nan)
            ).clip(0.2, 5.0)

    # --- Category 5: Hurst Exponent Features (12) ---

    @staticmethod
    def _hurst_rs(x: np.ndarray) -> float:
        """Rescaled range (R/S) Hurst exponent estimate.

        H > 0.5 = trending (form persists), H < 0.5 = mean-reverting.
        """
        n = len(x)
        if n < 4:
            return np.nan
        mean = np.mean(x)
        y = x - mean
        cumdev = np.cumsum(y)
        r = np.max(cumdev) - np.min(cumdev)
        s = np.std(x, ddof=1)
        if s == 0 or r == 0:
            return np.nan
        rs = r / s
        h = np.log(rs) / np.log(n)
        return np.clip(h, 0.2, 0.8)

    def _build_hurst_features(self, df: pd.DataFrame) -> None:
        """Build rolling Hurst exponent features via R/S analysis."""
        hw = self.hurst_window

        for stat in self.NICHE_STATS:
            for side in self.SIDES:
                col = f'{side}_{stat}'
                team_col = f'{side}_team'
                if col not in df.columns or team_col not in df.columns:
                    continue

                df[f'{side}_{stat}_hurst'] = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(
                        window=hw, min_periods=hw
                    ).apply(DynamicsFeatureEngineer._hurst_rs, raw=True)
                )

            # Hurst diff: home - away
            home_h = f'home_{stat}_hurst'
            away_h = f'away_{stat}_hurst'
            if home_h in df.columns and away_h in df.columns:
                df[f'{stat}_hurst_diff'] = df[home_h] - df[away_h]

    # --- Category 6: Damped Trend Features (12) ---

    @staticmethod
    def _holt_damped_trend(x: np.ndarray, alpha: float = 0.3,
                           beta: float = 0.1, phi: float = 0.9) -> np.ndarray:
        """Holt's exponential smoothing with damped trend.

        Returns the trend component array. Smoothed directional signal
        with mean-reversion damping (phi parameter).
        """
        n = len(x)
        trend = np.full(n, np.nan)
        if n < 2:
            return trend

        # Initialize
        level = x[0]
        b = x[1] - x[0]  # initial trend estimate
        trend[0] = b

        for t in range(1, n):
            if np.isnan(x[t]):
                trend[t] = np.nan
                continue
            new_level = alpha * x[t] + (1 - alpha) * (level + phi * b)
            b = beta * (new_level - level) + (1 - beta) * phi * b
            level = new_level
            trend[t] = np.clip(b, -2, 2)

        return trend

    def _build_damped_trend_features(self, df: pd.DataFrame) -> None:
        """Build Holt's damped trend features."""
        phi = self.damping_factor

        for stat in self.NICHE_STATS:
            for side in self.SIDES:
                col = f'{side}_{stat}'
                team_col = f'{side}_team'
                if col not in df.columns or team_col not in df.columns:
                    continue

                df[f'{side}_{stat}_damped_trend'] = df.groupby(team_col)[col].transform(
                    lambda x, _phi=phi: pd.Series(
                        DynamicsFeatureEngineer._holt_damped_trend(
                            x.shift(1).values, phi=_phi
                        ),
                        index=x.index,
                    )
                )

            # Damped trend diff: home - away
            home_dt = f'home_{stat}_damped_trend'
            away_dt = f'away_{stat}_damped_trend'
            if home_dt in df.columns and away_dt in df.columns:
                df[f'{stat}_damped_trend_diff'] = df[home_dt] - df[away_dt]
