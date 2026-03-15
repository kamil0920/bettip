"""
Spectral Feature Engineering — Frequency-Domain Analysis per Team

GBDTs see temporal patterns (EMAs, trends, entropy) in the time domain but cannot
detect cyclical behaviour in the frequency domain. Spectral entropy quantifies how
spread the power spectral density is — low values indicate periodic (predictable)
cycles while high values indicate noise.

This engineer adds frequency-domain features that complement the existing
dynamics (distributional), entropy (ordinal/sample), and window_ratio (multi-scale)
engineers.

Data: Loads match_stats.parquet (same source as DynamicsFeatureEngineer).
"""
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.signal import welch

from src.data_collection.match_stats_utils import normalize_match_stats_columns
from src.features.engineers.base import BaseFeatureEngineer
from src.leagues import EUROPEAN_LEAGUES

logger = logging.getLogger(__name__)


def _spectral_entropy(x: np.ndarray, min_periods: int = 10) -> float:
    """Normalized Shannon entropy of the power spectral density.

    Low = periodic signal (strong frequency components).
    High = noise-like (flat PSD, no dominant frequency).

    Returns value in [0, 1] or NaN if insufficient data.
    """
    x = x[~np.isnan(x)]
    if len(x) < min_periods or np.std(x) < 1e-10:
        return np.nan
    freqs, psd = welch(x, nperseg=min(len(x), 64))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    se = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    max_se = np.log2(len(psd_norm))
    return se / max_se if max_se > 0 else np.nan


def _dominant_period(x: np.ndarray, min_periods: int = 10) -> float:
    """Period of the strongest frequency component (excluding DC).

    Returns the number of samples per cycle at the peak PSD frequency,
    or NaN if the signal is too short or flat.
    """
    x = x[~np.isnan(x)]
    if len(x) < min_periods or np.std(x) < 1e-10:
        return np.nan
    freqs, psd = welch(x, nperseg=min(len(x), 64))
    if len(freqs) < 2 or freqs[1] == 0:
        return np.nan
    # Skip DC component (index 0)
    dominant_idx = np.argmax(psd[1:]) + 1
    return 1.0 / freqs[dominant_idx] if freqs[dominant_idx] > 0 else np.nan


class SpectralFeatureEngineer(BaseFeatureEngineer):
    """Frequency-domain features: spectral entropy and dominant period.

    Produces ~30 features:
    - Spectral entropy per side per stat (10) + diffs (5) + sums (5)
    - Dominant period per side per stat (10)

    Stats: fouls, shots, corners, cards, goals (5 stats x 2 sides = 10 base series)

    All features use shift(1) to prevent data leakage.
    """

    STATS = ['fouls', 'shots', 'corners', 'cards', 'goals']
    SIDES = ['home', 'away']

    def __init__(
        self,
        window: int = 15,
        min_periods: int = 10,
    ):
        self.window = window
        self.min_periods = min_periods
        self.data_dir = Path("data/01-raw")

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create spectral features from match stats."""
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
        """Build spectral entropy and dominant period features."""
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)

        min_p = self.min_periods
        w = self.window

        for stat in self.STATS:
            for side in self.SIDES:
                col = f'{side}_{stat}'
                if col not in df.columns:
                    continue
                team_col = f'{side}_team'
                if team_col not in df.columns:
                    continue

                # Spectral entropy: shift(1) prevents look-ahead bias
                se_col = f'{side}_{stat}_spectral_entropy'
                df[se_col] = (
                    df.groupby(team_col)[col]
                    .transform(
                        lambda x: x.shift(1)
                        .rolling(w, min_periods=min_p)
                        .apply(lambda v: _spectral_entropy(v, min_p), raw=True)
                    )
                ).clip(0, 1).fillna(0.5)

                # Dominant period: shift(1) prevents look-ahead bias
                dp_col = f'{side}_{stat}_dominant_period'
                df[dp_col] = (
                    df.groupby(team_col)[col]
                    .transform(
                        lambda x: x.shift(1)
                        .rolling(w, min_periods=min_p)
                        .apply(lambda v: _dominant_period(v, min_p), raw=True)
                    )
                ).clip(1, 50).fillna(5.0)

        # Cross-side features: diff and sum for spectral entropy
        for stat in self.STATS:
            h_se = f'home_{stat}_spectral_entropy'
            a_se = f'away_{stat}_spectral_entropy'
            if h_se in df.columns and a_se in df.columns:
                df[f'{stat}_spectral_entropy_diff'] = df[h_se] - df[a_se]
                df[f'{stat}_spectral_entropy_sum'] = (df[h_se] + df[a_se]) / 2

        n_features = len([c for c in df.columns if 'spectral' in c or 'dominant_period' in c])
        logger.info(f"SpectralFeatureEngineer: created {n_features} features")
        return df
