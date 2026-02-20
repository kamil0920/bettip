"""
Entropy Feature Engineering — Rolling Permutation Entropy & Sample Entropy per Team

GBDTs see levels (EMAs) and distribution shape (kurtosis, skewness from S30 dynamics)
but not **sequential complexity** — whether a team's stat series has structured ordinal
patterns or is random noise.

- **PE** measures ordinal pattern complexity (rank-based, scale-invariant). Low PE = predictable ordering.
- **SampEn** measures self-similarity at a tolerance (magnitude-based). Low SampEn = repeating patterns.

PE is already used as a static forecastability gate (experiments/forecastability_analysis.py),
but never as per-team rolling features. Correlation between PE and existing features
(kurtosis, volatility) is typically 0.3-0.5 — partially independent signals.

Data: Loads match_stats.parquet (same source as DynamicsFeatureEngineer).
"""
import logging
from collections import defaultdict
from math import factorial, log
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data_collection.match_stats_utils import normalize_match_stats_columns
from src.features.engineers.base import BaseFeatureEngineer
from src.leagues import EUROPEAN_LEAGUES

logger = logging.getLogger(__name__)


def _permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Normalized permutation entropy [0, 1] for a 1D array.

    0 = fully deterministic, 1 = completely random.
    Adapted from experiments/forecastability_analysis.py for rolling window use.
    """
    n = len(x)
    n_patterns = n - (order - 1) * delay

    if n_patterns < order:
        return np.nan

    # Count ordinal patterns
    pattern_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    for i in range(n_patterns):
        indices = [i + j * delay for j in range(order)]
        window = x[indices]
        pattern = tuple(np.argsort(np.argsort(window)))
        pattern_counts[pattern] += 1

    # Shannon entropy of pattern distribution
    total = sum(pattern_counts.values())
    probs = np.array([c / total for c in pattern_counts.values()])
    probs = probs[probs > 0]
    h = -np.sum(probs * np.log2(probs))

    # Normalize by maximum entropy (log2 of order!)
    h_max = log(factorial(order), 2)
    return h / h_max if h_max > 0 else 0.0


def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Sample entropy for a 1D array.

    Low = predictable (repeating patterns), high = random.
    Adapted from experiments/forecastability_analysis.py for rolling window use.
    """
    n = len(x)
    if n < m + 2:
        return np.nan

    std = np.std(x)
    if std == 0:
        return 0.0
    r = r_factor * std

    def _count_matches(template_len: int) -> int:
        count = 0
        templates = np.array([x[i:i + template_len] for i in range(n - template_len)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count

    a = _count_matches(m + 1)
    b = _count_matches(m)

    if b == 0:
        return np.nan

    return -log(a / b) if a > 0 else np.nan


class EntropyFeatureEngineer(BaseFeatureEngineer):
    """
    Generates rolling permutation entropy and sample entropy features.

    Produces 40 features across 2 categories:
    - PE (20): per-side (10) + diff (5) + sum (5)
    - SampEn (20): per-side (10) + diff (5) + sum (5)

    Stats: fouls, shots, corners, cards, goals (5 stats x 2 sides = 10 base series)

    All features use shift(1) to prevent data leakage.
    """

    STATS = ['fouls', 'shots', 'corners', 'cards', 'goals']
    SIDES = ['home', 'away']

    def __init__(
        self,
        window: int = 15,
        pe_order: int = 3,
        pe_delay: int = 1,
        sampen_m: int = 2,
        sampen_r_factor: float = 0.2,
        min_matches: int = 8,
    ):
        self.window = window
        self.pe_order = pe_order
        self.pe_delay = pe_delay
        self.sampen_m = sampen_m
        self.sampen_r_factor = sampen_r_factor
        self.min_matches = min_matches
        self.data_dir = Path("data/01-raw")

    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create entropy features from match stats."""
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
                if yellow in df.columns:
                    red = df.get(f'{side}_red_cards', 0)
                    df[col] = df[yellow].fillna(0) + pd.Series(red).fillna(0)
        return df

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all entropy features."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)
        df = self._derive_cards(df)

        self._build_pe_features(df)
        self._build_sampen_features(df)

        return df

    # --- Permutation Entropy (20 features) ---

    def _build_pe_features(self, df: pd.DataFrame) -> None:
        """Build rolling permutation entropy features."""
        window = self.window
        min_p = self.min_matches
        order = self.pe_order
        delay = self.pe_delay

        def _pe_window(x: np.ndarray) -> float:
            return _permutation_entropy(x, order=order, delay=delay)

        for stat in self.STATS:
            for side in self.SIDES:
                col = f'{side}_{stat}' if stat != 'goals' else f'{side}_goals'
                team_col = f'{side}_team'
                if col not in df.columns or team_col not in df.columns:
                    continue

                df[f'{side}_{stat}_pe'] = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(
                        window=window, min_periods=min_p
                    ).apply(_pe_window, raw=True)
                ).clip(0, 1)

            # Diff and sum
            home_pe = f'home_{stat}_pe'
            away_pe = f'away_{stat}_pe'
            if home_pe in df.columns and away_pe in df.columns:
                df[f'{stat}_pe_diff'] = df[home_pe] - df[away_pe]
                df[f'{stat}_pe_sum'] = df[home_pe] + df[away_pe]

    # --- Sample Entropy (20 features) ---

    def _build_sampen_features(self, df: pd.DataFrame) -> None:
        """Build rolling sample entropy features."""
        window = self.window
        min_p = self.min_matches
        m = self.sampen_m
        r_factor = self.sampen_r_factor

        def _sampen_window(x: np.ndarray) -> float:
            return _sample_entropy(x, m=m, r_factor=r_factor)

        for stat in self.STATS:
            for side in self.SIDES:
                col = f'{side}_{stat}' if stat != 'goals' else f'{side}_goals'
                team_col = f'{side}_team'
                if col not in df.columns or team_col not in df.columns:
                    continue

                df[f'{side}_{stat}_sampen'] = df.groupby(team_col)[col].transform(
                    lambda x: x.shift(1).rolling(
                        window=window, min_periods=min_p
                    ).apply(_sampen_window, raw=True)
                ).clip(0, 5)

            # Diff and sum
            home_se = f'home_{stat}_sampen'
            away_se = f'away_{stat}_sampen'
            if home_se in df.columns and away_se in df.columns:
                df[f'{stat}_sampen_diff'] = df[home_se] - df[away_se]
                df[f'{stat}_sampen_sum'] = df[home_se] + df[away_se]
